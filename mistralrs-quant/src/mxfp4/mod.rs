use std::{
    collections::HashMap,
    sync::{atomic::AtomicUsize, Arc},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_hadd_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm_add_ss, _mm_cvtss_f32, _mm256_fmadd_ps, _mm256_setzero_ps,
};

use candle_core::{DType, Device, Result, Storage, Tensor};
use rayon::prelude::*;
use safetensors::tensor::Dtype;

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::mxfp4_stream::{MxFp4StreamCache, MxFp4StreamKey, MxFp4StreamRange};
use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, Shard, ShardedVarBuilder, UqffReader, UqffTensor,
};

#[cfg(feature = "cuda")]
pub(crate) mod ffi;
#[cfg(feature = "metal")]
pub(crate) mod metal_ops;
#[cfg(feature = "cuda")]
pub(crate) mod ops;

/// MXFP4 block size (32 elements per scale)
pub const MXFP4_BLOCK_SIZE: usize = 32;

pub(crate) const N_BITS: usize = 4;

#[derive(Debug)]
pub struct MXFP4Layer {
    /// Packed FP4 weights: [N, K/2] or [num_experts, N, K/2]
    /// Each byte contains 2 FP4 values (low nibble = k, high nibble = k+1)
    #[allow(dead_code)]
    blocks: Tensor,
    /// E8M0 scales: [N, K/32] or [num_experts, N, K/32]
    /// Each byte is an 8-bit exponent with bias 127
    scales: Tensor,
    /// Optional bias: [N] or [num_experts, N]
    #[allow(dead_code)]
    bias: Option<Tensor>,
}

/// File-backed GPT-OSS MXFP4 expert projection.
///
/// The packed expert bank remains file-backed in GGUF. gather_forward_raw loads
/// only routed experts through the shared bounded stream cache and decodes them
/// into a temporary output buffer.
#[derive(Debug)]
pub struct MxFp4StreamingExpertLayer {
    raw_weights: Vec<String>,
    num_experts: usize,
    component_out_dim: usize,
    in_dim: usize,
    out_dim: usize,
    bias: Option<Tensor>,
    bias_cpu: Option<Arc<Vec<f32>>>,
    expert_ranges: Vec<MxFp4StreamRange>,
    cache: Arc<MxFp4StreamCache>,
}

impl MxFp4StreamingExpertLayer {
    pub(crate) fn from_gguf(
        archive: Arc<crate::GgufArchive>,
        raw_weights: Vec<String>,
        num_experts: usize,
        in_dim: usize,
        out_dim: usize,
        bias: Option<Tensor>,
        cache: Arc<MxFp4StreamCache>,
    ) -> Result<Self> {
        if raw_weights.is_empty() || raw_weights.len() > 2 {
            candle_core::bail!("GPT-OSS MXFP4 streaming expects one or two raw expert tensors");
        }
        if raw_weights.len() == 2 && !out_dim.is_multiple_of(2) {
            candle_core::bail!(
                "GPT-OSS gate/up streamed projection output dimension must be even, got {out_dim}"
            );
        }

        let component_out_dim = if raw_weights.len() == 2 {
            out_dim / 2
        } else {
            out_dim
        };

        let mut expert_ranges = Vec::with_capacity(raw_weights.len());
        for name in &raw_weights {
            let info = archive.tensor_info(name)?;
            if info.dtype().raw() != 39 {
                candle_core::bail!(
                    "GGUF tensor `{name}` has dtype {}, expected MXFP4 dtype 39",
                    info.dtype().raw()
                );
            }
            let Some(last) = info.shape().last() else {
                candle_core::bail!("GGUF MXFP4 tensor `{name}` has no dimensions");
            };
            if !last.is_multiple_of(MXFP4_BLOCK_SIZE) {
                candle_core::bail!(
                    "GGUF MXFP4 tensor `{name}` last dimension {last} is not divisible by {MXFP4_BLOCK_SIZE}"
                );
            }
            let shape = info.shape();
            if shape.len() != 3
                || shape[0] != num_experts
                || shape[1] != component_out_dim
                || shape[2] != in_dim
            {
                candle_core::bail!(
                    "GPT-OSS MXFP4 streaming tensor {name} has shape {shape:?}, expected [{num_experts}, {component_out_dim}, {in_dim}]"
                );
            }
            if !in_dim.is_multiple_of(MXFP4_BLOCK_SIZE) {
                candle_core::bail!(
                    "GPT-OSS MXFP4 streaming tensor {name} has K {in_dim}, not divisible by block size {MXFP4_BLOCK_SIZE}"
                );
            }
            let expected_bytes = num_experts
                .checked_mul(component_out_dim)
                .and_then(|v| v.checked_mul(in_dim / MXFP4_BLOCK_SIZE))
                .and_then(|v| v.checked_mul(MXFP4_BLOCK_SIZE / 2 + 1))
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "GPT-OSS MXFP4 streaming byte-size overflow for {name}"
                    ))
                })?;
            let actual_bytes = archive.tensor_data(name)?.bytes().len();
            if actual_bytes != expected_bytes {
                candle_core::bail!(
                    "GPT-OSS MXFP4 streaming tensor {name} has {actual_bytes} bytes, expected {expected_bytes}"
                );
            }

            let base = info
                .data_range()
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "GPT-OSS MXFP4 tensor {name} has no data range"
                    ))
                })?
                .start;
            let file_len = archive
                .shards()
                .get(info.shard_index())
                .ok_or_else(|| {
                    candle_core::Error::Msg("GPT-OSS MXFP4 shard index out of range".into())
                })?
                .file_len();
            expert_ranges.push(MxFp4StreamRange {
                shard: info.shard_index(),
                offset: u64::try_from(base).map_err(|_| {
                    candle_core::Error::Msg("GPT-OSS MXFP4 base offset exceeds u64".into())
                })?,
                len: expected_bytes / num_experts,
                file_len,
            });
        }

        let bias_cpu = if let Some(bias) = &bias {
            if bias.dims() != [num_experts, out_dim] {
                candle_core::bail!(
                    "GPT-OSS MXFP4 streaming bias has shape {:?}, expected [{num_experts}, {out_dim}]",
                    bias.dims()
                );
            }
            let values = bias
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = num_experts * out_dim;
            if values.len() != expected {
                candle_core::bail!(
                    "GPT-OSS MXFP4 streaming bias has {} elements, expected {}",
                    values.len(),
                    expected
                );
            }
            Some(Arc::new(values))
        } else {
            None
        };

        Ok(Self {
            raw_weights,
            num_experts,
            component_out_dim,
            in_dim,
            out_dim,
            bias,
            bias_cpu,
            expert_ranges,
            cache,
        })
    }


    #[inline(always)]
    fn raw_expert_range(
        &self,
        weight_idx: usize,
        expert_idx: usize,
    ) -> Result<MxFp4StreamRange> {
        let base = *self.expert_ranges.get(weight_idx).ok_or_else(|| {
            candle_core::Error::Msg("invalid streamed MXFP4 weight index".into())
        })?;
        if expert_idx >= self.num_experts {
            candle_core::bail!(
                "GPT-OSS MXFP4 expert index {expert_idx} out of range for {} experts",
                self.num_experts
            );
        }
        let byte_offset = expert_idx
            .checked_mul(base.len)
            .ok_or_else(|| candle_core::Error::Msg("GPT-OSS MXFP4 expert offset overflow".into()))?;
        let offset = base
            .offset
            .checked_add(u64::try_from(byte_offset).map_err(|_| {
                candle_core::Error::Msg("GPT-OSS MXFP4 expert offset exceeds u64".into())
            })?)
            .ok_or_else(|| candle_core::Error::Msg("GPT-OSS MXFP4 expert offset overflow".into()))?;
        Ok(MxFp4StreamRange { offset, ..base })
    }

    #[inline(always)]
    fn dot_block(x: &[f32], w: &[f32], kernel: u8) -> f32 {
        debug_assert_eq!(x.len(), MXFP4_BLOCK_SIZE);
        debug_assert_eq!(w.len(), MXFP4_BLOCK_SIZE);

        #[cfg(target_arch = "x86_64")]
        {
            if kernel >= 2 {
                return unsafe { Self::dot_block_avx2_fma(x, w) };
            }
            if kernel == 1 {
                return unsafe { Self::dot_block_avx2(x, w) };
            }
        }

        let mut acc = 0f32;
        for i in 0..MXFP4_BLOCK_SIZE {
            acc += x[i] * w[i];
        }
        acc
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot_block_avx2(x: &[f32], w: &[f32]) -> f32 {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        a0 = _mm256_add_ps(
            a0,
            _mm256_mul_ps(
                _mm256_loadu_ps(x.as_ptr()),
                _mm256_loadu_ps(w.as_ptr()),
            ),
        );
        a1 = _mm256_add_ps(
            a1,
            _mm256_mul_ps(
                _mm256_loadu_ps(x.as_ptr().add(8)),
                _mm256_loadu_ps(w.as_ptr().add(8)),
            ),
        );
        a2 = _mm256_add_ps(
            a2,
            _mm256_mul_ps(
                _mm256_loadu_ps(x.as_ptr().add(16)),
                _mm256_loadu_ps(w.as_ptr().add(16)),
            ),
        );
        a3 = _mm256_add_ps(
            a3,
            _mm256_mul_ps(
                _mm256_loadu_ps(x.as_ptr().add(24)),
                _mm256_loadu_ps(w.as_ptr().add(24)),
            ),
        );
        Self::hsum4(a0, a1, a2, a3)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot_block_avx2_fma(x: &[f32], w: &[f32]) -> f32 {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr()),
            _mm256_loadu_ps(w.as_ptr()),
            a0,
        );
        a1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(8)),
            _mm256_loadu_ps(w.as_ptr().add(8)),
            a1,
        );
        a2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(16)),
            _mm256_loadu_ps(w.as_ptr().add(16)),
            a2,
        );
        a3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(x.as_ptr().add(24)),
            _mm256_loadu_ps(w.as_ptr().add(24)),
            a3,
        );
        Self::hsum4(a0, a1, a2, a3)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn hsum4(a0: __m256, a1: __m256, a2: __m256, a3: __m256) -> f32 {
        let s01 = _mm256_add_ps(a0, a1);
        let s23 = _mm256_add_ps(a2, a3);
        let sum = _mm256_add_ps(s01, s23);
        let h1 = _mm256_hadd_ps(sum, sum);
        let h2 = _mm256_hadd_ps(h1, h1);
        let lo = _mm256_castps256_ps128(h2);
        let hi = _mm256_extractf128_ps(h2, 1);
        _mm_cvtss_f32(_mm_add_ss(lo, hi))
    }

    #[inline(always)]
    fn dot_kernel() -> u8 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                if std::is_x86_feature_detected!("fma") {
                    return 2;
                }
                return 1;
            }
        }
        0
    }

    const GEMM_MIN_ROUTES: usize = 8;

    fn decode_expert_f32(
        raw_expert: &[u8],
        out_rows: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let mut weights = vec![0f32; out_rows * in_dim];

        weights
            .par_chunks_mut(in_dim)
            .enumerate()
            .for_each(|(row, dst)| {
                let row_start = row * row_bytes;
                for block_idx in 0..blocks_per_row {
                    let block_start =
                        row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                    let dequant = &MXFP4Layer::DEQUANT_LUT[raw_expert[block_start] as usize];
                    let packed =
                        &raw_expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                    let col_start = block_idx * MXFP4_BLOCK_SIZE;
                    for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                        let packed_byte = packed[byte_idx];
                        dst[col_start + byte_idx] =
                            dequant[(packed_byte & 0x0f) as usize];
                        dst[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx] =
                            dequant[(packed_byte >> 4) as usize];
                    }
                }
            });

        weights
    }

    fn gemm_routes(
        x_data: &[f32],
        route_x_offsets: &[usize],
        raw_expert: &[u8],
        out_rows: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let route_count = route_x_offsets.len();
        if route_count < Self::GEMM_MIN_ROUTES {
            return Err(candle_core::Error::Msg(
                "route count below streamed GEMM threshold".into(),
            ));
        }

        let mut x_routes = Vec::with_capacity(route_count * in_dim);
        for &x_offset in route_x_offsets {
            x_routes.extend_from_slice(&x_data[x_offset..x_offset + in_dim]);
        }

        let weights = Self::decode_expert_f32(raw_expert, out_rows, in_dim);
        let x = Tensor::from_vec(x_routes, (route_count, in_dim), &Device::Cpu)?;
        let w = Tensor::from_vec(weights, (out_rows, in_dim), &Device::Cpu)?;
        let y = x.matmul(&w.t()?)?;
        y.flatten_all()?.to_vec1::<f32>()
    }

    #[inline(always)]
    fn dot_streamed_row_group(
        x_data: &[f32],
        x_offsets: &[usize],
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
        route_count: usize,
    ) -> Option<[f32; 4]> {
        if route_count < 2 || route_count > 4 {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                if std::is_x86_feature_detected!("fma") {
                    return Some(unsafe {
                        Self::dot_streamed_row_group_avx2_fma(
                            x_data,
                            x_offsets,
                            raw_expert,
                            row,
                            in_dim,
                            route_count,
                        )
                    });
                }
                return Some(unsafe {
                    Self::dot_streamed_row_group_avx2(
                        x_data,
                        x_offsets,
                        raw_expert,
                        row,
                        in_dim,
                        route_count,
                    )
                });
            }
        }
        None
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_streamed_row_group_avx2(
        x_data: &[f32],
        x_offsets: &[usize],
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
        route_count: usize,
    ) -> [f32; 4] {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let row_start = row * row_bytes;
        let mut acc = [_mm256_setzero_ps(); 4 * 4];
        let mut w_block = [0f32; MXFP4_BLOCK_SIZE];

        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let dequant = &MXFP4Layer::DEQUANT_LUT[raw_expert[block_start] as usize];
            let packed =
                &raw_expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                let packed_byte = packed[byte_idx];
                w_block[byte_idx] = dequant[(packed_byte & 0x0f) as usize];
                w_block[MXFP4_BLOCK_SIZE / 2 + byte_idx] =
                    dequant[(packed_byte >> 4) as usize];
            }

            let x_start = block_idx * MXFP4_BLOCK_SIZE;
            for route in 0..route_count {
                let x_row = &x_data[x_offsets[route]..x_offsets[route] + in_dim];
                let base = route * 4;
                acc[base] = _mm256_add_ps(
                    acc[base],
                    _mm256_mul_ps(
                        _mm256_loadu_ps(x_row.as_ptr().add(x_start)),
                        _mm256_loadu_ps(w_block.as_ptr()),
                    ),
                );
                acc[base + 1] = _mm256_add_ps(
                    acc[base + 1],
                    _mm256_mul_ps(
                        _mm256_loadu_ps(x_row.as_ptr().add(x_start + 8)),
                        _mm256_loadu_ps(w_block.as_ptr().add(8)),
                    ),
                );
                acc[base + 2] = _mm256_add_ps(
                    acc[base + 2],
                    _mm256_mul_ps(
                        _mm256_loadu_ps(x_row.as_ptr().add(x_start + 16)),
                        _mm256_loadu_ps(w_block.as_ptr().add(16)),
                    ),
                );
                acc[base + 3] = _mm256_add_ps(
                    acc[base + 3],
                    _mm256_mul_ps(
                        _mm256_loadu_ps(x_row.as_ptr().add(x_start + 24)),
                        _mm256_loadu_ps(w_block.as_ptr().add(24)),
                    ),
                );
            }
        }

        let mut out = [0.0f32; 4];
        for route in 0..route_count {
            let base = route * 4;
            out[route] = Self::hsum4(
                acc[base],
                acc[base + 1],
                acc[base + 2],
                acc[base + 3],
            );
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_streamed_row_group_avx2_fma(
        x_data: &[f32],
        x_offsets: &[usize],
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
        route_count: usize,
    ) -> [f32; 4] {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let row_start = row * row_bytes;
        let mut acc = [_mm256_setzero_ps(); 4 * 4];
        let mut w_block = [0f32; MXFP4_BLOCK_SIZE];

        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let dequant = &MXFP4Layer::DEQUANT_LUT[raw_expert[block_start] as usize];
            let packed =
                &raw_expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                let packed_byte = packed[byte_idx];
                w_block[byte_idx] = dequant[(packed_byte & 0x0f) as usize];
                w_block[MXFP4_BLOCK_SIZE / 2 + byte_idx] =
                    dequant[(packed_byte >> 4) as usize];
            }

            let x_start = block_idx * MXFP4_BLOCK_SIZE;
            for route in 0..route_count {
                let x_row = &x_data[x_offsets[route]..x_offsets[route] + in_dim];
                let base = route * 4;
                acc[base] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row.as_ptr().add(x_start)),
                    _mm256_loadu_ps(w_block.as_ptr()),
                    acc[base],
                );
                acc[base + 1] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row.as_ptr().add(x_start + 8)),
                    _mm256_loadu_ps(w_block.as_ptr().add(8)),
                    acc[base + 1],
                );
                acc[base + 2] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row.as_ptr().add(x_start + 16)),
                    _mm256_loadu_ps(w_block.as_ptr().add(16)),
                    acc[base + 2],
                );
                acc[base + 3] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x_row.as_ptr().add(x_start + 24)),
                    _mm256_loadu_ps(w_block.as_ptr().add(24)),
                    acc[base + 3],
                );
            }
        }

        let mut out = [0.0f32; 4];
        for route in 0..route_count {
            let base = route * 4;
            out[route] = Self::hsum4(
                acc[base],
                acc[base + 1],
                acc[base + 2],
                acc[base + 3],
            );
        }
        out
    }

    #[inline(always)]
    fn dot_streamed_row(x: &[f32], raw_expert: &[u8], row: usize, kernel: u8) -> f32 {
        debug_assert!(x.len() >= MXFP4_BLOCK_SIZE);
        let blocks_per_row = x.len() / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let row_start = row * row_bytes;

        #[cfg(target_arch = "x86_64")]
        {
            if kernel >= 2 {
                return unsafe { Self::dot_streamed_row_avx2_fma(x, raw_expert, row_start, blocks_per_row) };
            }
            if kernel == 1 {
                return unsafe { Self::dot_streamed_row_avx2(x, raw_expert, row_start, blocks_per_row) };
            }
        }

        let mut acc = 0f32;
        let mut w_block = [0f32; MXFP4_BLOCK_SIZE];
        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let dequant = &MXFP4Layer::DEQUANT_LUT[raw_expert[block_start] as usize];
            let packed =
                &raw_expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                let packed_byte = packed[byte_idx];
                w_block[byte_idx] = dequant[(packed_byte & 0x0f) as usize];
                w_block[MXFP4_BLOCK_SIZE / 2 + byte_idx] =
                    dequant[(packed_byte >> 4) as usize];
            }
            let col_start = block_idx * MXFP4_BLOCK_SIZE;
            acc += Self::dot_block(
                &x[col_start..col_start + MXFP4_BLOCK_SIZE],
                &w_block,
                0,
            );
        }
        acc
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_streamed_row_avx2(
        x: &[f32],
        raw_expert: &[u8],
        row_start: usize,
        blocks_per_row: usize,
    ) -> f32 {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut w_block = [0f32; MXFP4_BLOCK_SIZE];

        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let dequant = &MXFP4Layer::DEQUANT_LUT[raw_expert[block_start] as usize];
            let packed =
                &raw_expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                let packed_byte = packed[byte_idx];
                w_block[byte_idx] = dequant[(packed_byte & 0x0f) as usize];
                w_block[MXFP4_BLOCK_SIZE / 2 + byte_idx] =
                    dequant[(packed_byte >> 4) as usize];
            }
            let x_start = block_idx * MXFP4_BLOCK_SIZE;
            a0 = _mm256_add_ps(
                a0,
                _mm256_mul_ps(
                    _mm256_loadu_ps(x.as_ptr().add(x_start)),
                    _mm256_loadu_ps(w_block.as_ptr()),
                ),
            );
            a1 = _mm256_add_ps(
                a1,
                _mm256_mul_ps(
                    _mm256_loadu_ps(x.as_ptr().add(x_start + 8)),
                    _mm256_loadu_ps(w_block.as_ptr().add(8)),
                ),
            );
            a2 = _mm256_add_ps(
                a2,
                _mm256_mul_ps(
                    _mm256_loadu_ps(x.as_ptr().add(x_start + 16)),
                    _mm256_loadu_ps(w_block.as_ptr().add(16)),
                ),
            );
            a3 = _mm256_add_ps(
                a3,
                _mm256_mul_ps(
                    _mm256_loadu_ps(x.as_ptr().add(x_start + 24)),
                    _mm256_loadu_ps(w_block.as_ptr().add(24)),
                ),
            );
        }
        Self::hsum4(a0, a1, a2, a3)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_streamed_row_avx2_fma(
        x: &[f32],
        raw_expert: &[u8],
        row_start: usize,
        blocks_per_row: usize,
    ) -> f32 {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut w_block = [0f32; MXFP4_BLOCK_SIZE];

        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let dequant = &MXFP4Layer::DEQUANT_LUT[raw_expert[block_start] as usize];
            let packed =
                &raw_expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                let packed_byte = packed[byte_idx];
                w_block[byte_idx] = dequant[(packed_byte & 0x0f) as usize];
                w_block[MXFP4_BLOCK_SIZE / 2 + byte_idx] =
                    dequant[(packed_byte >> 4) as usize];
            }
            let x_start = block_idx * MXFP4_BLOCK_SIZE;
            a0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(x.as_ptr().add(x_start)),
                _mm256_loadu_ps(w_block.as_ptr()),
                a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(x.as_ptr().add(x_start + 8)),
                _mm256_loadu_ps(w_block.as_ptr().add(8)),
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(x.as_ptr().add(x_start + 16)),
                _mm256_loadu_ps(w_block.as_ptr().add(16)),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(x.as_ptr().add(x_start + 24)),
                _mm256_loadu_ps(w_block.as_ptr().add(24)),
                a3,
            );
        }
        Self::hsum4(a0, a1, a2, a3)
    }

    fn dot_row(
        x: &[f32],
        raw_expert: &[u8],
        row: usize,
        out_row: &mut [f32],
        out_col: usize,
    ) {
        let blocks_per_row = x.len() / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let row_start = row * row_bytes;
        let mut acc = 0f32;

        for block_idx in 0..blocks_per_row {
            let start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let end = start + (MXFP4_BLOCK_SIZE / 2 + 1);
            let block = &raw_expert[start..end];
            let dequant = &MXFP4Layer::DEQUANT_LUT[block[0] as usize];
            let packed = &block[1..];

            let col_start = block_idx * MXFP4_BLOCK_SIZE;
            // GGML MXFP4 uses split-half packing: low nibbles are elements 0..15,
            // high nibbles are elements 16..31.
            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                let packed_byte = packed[byte_idx];
                acc += x[col_start + byte_idx]
                    * dequant[(packed_byte & 0x0f) as usize];
                acc += x[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx]
                    * dequant[(packed_byte >> 4) as usize];
            }
        }

        out_row[out_col] += acc;
    }
}

impl QuantMethod for MxFp4StreamingExpertLayer {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("MxFp4StreamingExpertLayer must be constructed from a GGUF archive")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        candle_core::bail!(
            "{} keeps expert weights file-backed and supports gather_forward only",
            self.name()
        )
    }

    fn forward_raw(&self, _a: &Tensor) -> Result<Tensor> {
        candle_core::bail!(
            "{} is a routed expert layer; use gather_forward",
            self.name()
        )
    }

    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims();
        let index_dims = indices.dims();
        if index_dims.len() != 2 {
            candle_core::bail!(
                "GPT-OSS MXFP4 streaming expects rank-2 indices, got rank {}",
                index_dims.len()
            );
        }

        let (num_tokens, topk, k, x_has_topk) = match x_dims {
            [tokens, cols] => {
                if *tokens != index_dims[0] {
                    candle_core::bail!(
                        "GPT-OSS MXFP4 streaming input and index token counts do not agree"
                    );
                }
                (*tokens, index_dims[1], *cols, false)
            }
            [tokens, x_topk, cols] => {
                // The routed MoE contract allows either one shared input row per
                // token ([tokens, 1, hidden]) or one input row per routed slot
                // ([tokens, topk, hidden]).  GPT-OSS passes the former while
                // topk_ids has one expert index for every route.
                if *tokens != index_dims[0] {
                    candle_core::bail!(
                        "GPT-OSS MXFP4 streaming input and index token counts do not agree"
                    );
                }
                if *x_topk != 1 && *x_topk != index_dims[1] {
                    candle_core::bail!(
                        "GPT-OSS MXFP4 streaming input route dimension {} does not match top-k {}",
                        x_topk,
                        index_dims[1]
                    );
                }
                (*tokens, index_dims[1], *cols, *x_topk != 1)
            }
            _ => candle_core::bail!(
                "GPT-OSS MXFP4 streaming expects rank-2 or rank-3 input, got rank {}",
                x_dims.len()
            ),
        };

        if k != self.in_dim || !k.is_multiple_of(MXFP4_BLOCK_SIZE) {
            candle_core::bail!(
                "GPT-OSS MXFP4 streaming input K {k} does not match expected {}",
                self.in_dim
            );
        }

        let x_cpu = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let x_data = x_cpu.flatten_all()?.to_vec1::<f32>()?;
        let indices_cpu = indices.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;
        let indices_data = indices_cpu.flatten_all()?.to_vec1::<u32>()?;

        let mut routes_by_expert = vec![Vec::<usize>::new(); self.num_experts];
        for (route, &expert) in indices_data.iter().enumerate() {
            let expert = expert as usize;
            if expert >= self.num_experts {
                candle_core::bail!(
                    "GPT-OSS MXFP4 expert index {} out of range for {} experts",
                    expert,
                    self.num_experts
                );
            }
            routes_by_expert[expert].push(route);
        }
        let mut experts = routes_by_expert
            .iter()
            .enumerate()
            .filter_map(|(expert, routes)| (!routes.is_empty()).then_some(expert))
            .collect::<Vec<_>>();
        // Load low-frequency experts first so the per-source cache quota retains
        // the most frequently routed experts at the end of the pass.
        experts.sort_unstable_by_key(|&expert| (routes_by_expert[expert].len(), expert));

        let mut requests = Vec::with_capacity(experts.len() * self.raw_weights.len());
        for weight_idx in 0..self.raw_weights.len() {
            for &expert_idx in &experts {
                let range = self.raw_expert_range(weight_idx, expert_idx)?;
                requests.push((
                    MxFp4StreamKey {
                        source: self.raw_weights[weight_idx].clone(),
                        expert_index: expert_idx,
                    },
                    range,
                ));
            }
        }

        let mut pending = if self.cache.overlap() {
            Some(self.cache.prefetch(&requests)?)
        } else {
            None
        };
        let mut expert_data = HashMap::with_capacity(requests.len());
        if pending.is_none() {
            for (key, range) in requests {
                expert_data.insert(key.clone(), self.cache.load(&key, range)?);
            }
        }

        let mut output = vec![0f32; num_tokens * topk * self.out_dim];
        let kernel = Self::dot_kernel();

        // Compute each routed expert once over all of its route rows. The previous
        // implementation parallelized over route rows and decoded the full expert
        // matrix independently for every route, which is especially expensive for
        // prompt batches where the same expert can serve multiple tokens.
        //
        // The grouped path below parallelizes over output rows, decodes each MXFP4
        // weight block once per row, and reuses it across all matching routes..
        if self.raw_weights.len() > 1 {
            for component in 0..self.raw_weights.len() {
                for &expert_idx in &experts {
                    let key = MxFp4StreamKey {
                        source: self.raw_weights[component].clone(),
                        expert_index: expert_idx,
                    };
                    if let Some(pending_map) = pending.as_mut() {
                        if !expert_data.contains_key(&key) {
                            let handle = pending_map.remove(&key).ok_or_else(|| {
                                candle_core::Error::Msg(
                                    "GPT-OSS MXFP4 streamed expert request was not scheduled"
                                        .to_string(),
                                )
                            })?;
                            let data = self.cache.resolve(&key, handle)?;
                            expert_data.insert(key.clone(), data);
                        }
                    }
                    let expert = expert_data.get(&key).ok_or_else(|| {
                        candle_core::Error::Msg(
                            "GPT-OSS MXFP4 streamed expert was not loaded".into(),
                        )
                    })?;
                    let routes = routes_by_expert.get(&expert_idx).ok_or_else(|| {
                        candle_core::Error::Msg(
                            "GPT-OSS MXFP4 route table lost a selected expert".into(),
                        )
                    })?;
                    let route_x_offsets: Vec<usize> = routes
                        .iter()
                        .map(|&route_row| {
                            if x_has_topk {
                                route_row * self.in_dim
                            } else {
                                (route_row / topk) * self.in_dim
                            }
                        })
                        .collect();

                    let blocks_per_row = self.in_dim / MXFP4_BLOCK_SIZE;
                    let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                    let route_count = routes.len();

                    if route_count >= Self::GEMM_MIN_ROUTES {
                        let result =
                            Self::gemm_routes(&x_data, &route_x_offsets, expert, self.component_out_dim, self.in_dim)?;
                        for (route_idx, &route_row) in routes.iter().enumerate() {
                            let out_row = &mut output
                                [route_row * self.out_dim..(route_row + 1) * self.out_dim];
                            let base = route_idx * self.component_out_dim;
                            for row in 0..self.component_out_dim {
                                out_row[row * 2 + component] += result[base + row];
                            }
                        }
                        continue;
                    }

                if route_count == 1 {
                        let route_row = routes[0];
                        let x_offset = route_x_offsets[0];
                        let x_row = &x_data[x_offset..x_offset + self.in_dim];
                        let out_row =
                            &mut output[route_row * self.out_dim..(route_row + 1) * self.out_dim];
                        out_row
                            .par_chunks_mut(2)
                            .enumerate()
                            .for_each(|(row, pair)| {
                                pair[component] +=
                                    Self::dot_streamed_row(x_row, expert, row, kernel);
                            });
                        continue;
                    }

                    let mut partial = vec![0f32; self.component_out_dim * route_count];

                    partial
                        .par_chunks_mut(route_count)
                        .enumerate()
                        .for_each(|(row, accs)| {
                            if let Some(grouped) = Self::dot_streamed_row_group(
                                &x_data,
                                &route_x_offsets,
                                expert,
                                row,
                                self.in_dim,
                                route_count,
                            ) {
                                accs[..route_count].copy_from_slice(&grouped[..route_count]);
                                return;
                            }

                            let row_start = row * row_bytes;
                            for block_idx in 0..blocks_per_row {
                                let block_start =
                                    row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                                let scale = expert[block_start] as usize;
                                let dequant = &MXFP4Layer::DEQUANT_LUT[scale];
                                let packed =
                                    &expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                let col_start = block_idx * MXFP4_BLOCK_SIZE;
                                let mut w_block = [0f32; MXFP4_BLOCK_SIZE];
                                for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                    let packed_byte = packed[byte_idx];
                                    w_block[byte_idx] = dequant[(packed_byte & 0x0f) as usize];
                                    w_block[MXFP4_BLOCK_SIZE / 2 + byte_idx] =
                                        dequant[(packed_byte >> 4) as usize];
                                }

                                for (route_idx, &x_offset) in route_x_offsets.iter().enumerate() {
                                    let x_row = &x_data[x_offset..x_offset + self.in_dim];
                                    let x_block = &x_row[col_start..col_start + MXFP4_BLOCK_SIZE];
                                    accs[route_idx] +=
                                        Self::dot_block(x_block, &w_block, kernel);
                                }
                            }
                        });

                    for (route_idx, &route_row) in routes.iter().enumerate() {
                        let out_row = &mut output
                            [route_row * self.out_dim..(route_row + 1) * self.out_dim];
                        for row in 0..self.component_out_dim {
                            out_row[row * 2 + component] +=
                                partial[row * route_count + route_idx];
                        }
                    }
                }
            }
        } else {
            let component = 0usize;
            for &expert_idx in &experts {
                let key = MxFp4StreamKey {
                    source: self.raw_weights[component].clone(),
                    expert_index: expert_idx,
                };
                if let Some(pending_map) = pending.as_mut() {
                    if !expert_data.contains_key(&key) {
                        let handle = pending_map.remove(&key).ok_or_else(|| {
                            candle_core::Error::Msg(
                                "GPT-OSS MXFP4 streamed expert request was not scheduled"
                                    .to_string(),
                            )
                        })?;
                        let data = self.cache.resolve(&key, handle)?;
                        expert_data.insert(key.clone(), data);
                    }
                }
                let expert = expert_data.get(&key).ok_or_else(|| {
                    candle_core::Error::Msg(
                        "GPT-OSS MXFP4 streamed expert was not loaded".into(),
                    )
                })?;
                let routes = routes_by_expert.get(&expert_idx).ok_or_else(|| {
                    candle_core::Error::Msg(
                        "GPT-OSS MXFP4 route table lost a selected expert".into(),
                    )
                })?;
                let route_x_offsets: Vec<usize> = routes
                    .iter()
                    .map(|&route_row| {
                        if x_has_topk {
                            route_row * self.in_dim
                        } else {
                            (route_row / topk) * self.in_dim
                        }
                    })
                    .collect();

                let blocks_per_row = self.in_dim / MXFP4_BLOCK_SIZE;
                let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                let route_count = routes.len();

                if route_count == 1 {
                    let route_row = routes[0];
                    let x_offset = route_x_offsets[0];
                    let x_row = &x_data[x_offset..x_offset + self.in_dim];
                    let out_row =
                        &mut output[route_row * self.out_dim..(route_row + 1) * self.out_dim];
                    out_row
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(row, value)| {
                            *value += Self::dot_streamed_row(x_row, expert, row, kernel);
                        });
                    continue;
                }

                let mut partial = vec![0f32; self.component_out_dim * route_count];

                partial
                    .par_chunks_mut(route_count)
                    .enumerate()
                    .for_each(|(row, accs)| {
                        if let Some(grouped) = Self::dot_streamed_row_group(
                            &x_data,
                            &route_x_offsets,
                            expert,
                            row,
                            self.in_dim,
                            route_count,
                        ) {
                            accs[..route_count].copy_from_slice(&grouped[..route_count]);
                            return;
                        }

                        let row_start = row * row_bytes;
                        for block_idx in 0..blocks_per_row {
                            let block_start =
                                row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                            let scale = expert[block_start] as usize;
                            let dequant = &MXFP4Layer::DEQUANT_LUT[scale];
                            let packed =
                                &expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                            let col_start = block_idx * MXFP4_BLOCK_SIZE;
                            let mut w_block = [0f32; MXFP4_BLOCK_SIZE];
                            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                let packed_byte = packed[byte_idx];
                                w_block[byte_idx] = dequant[(packed_byte & 0x0f) as usize];
                                w_block[MXFP4_BLOCK_SIZE / 2 + byte_idx] =
                                    dequant[(packed_byte >> 4) as usize];
                            }

                            for (route_idx, &x_offset) in route_x_offsets.iter().enumerate() {
                                let x_row = &x_data[x_offset..x_offset + self.in_dim];
                                let x_block = &x_row[col_start..col_start + MXFP4_BLOCK_SIZE];
                                accs[route_idx] += Self::dot_block(
                                    x_block,
                                    &w_block,
                                    kernel,
                                );
                            }
                        }
                    });

                for (route_idx, &route_row) in routes.iter().enumerate() {
                    let out_row = &mut output
                        [route_row * self.out_dim..(route_row + 1) * self.out_dim];
                    for row in 0..self.component_out_dim {
                        out_row[row] += partial[row * route_count + route_idx];
                    }
                }
            }
        }
        if let Some(bias_data) = &self.bias_cpu {
            output
                .par_chunks_mut(self.out_dim)
                .enumerate()
                .for_each(|(route_row, out_row)| {
                    let expert_idx = indices_data[route_row] as usize;
                    let bias_offset = expert_idx * self.out_dim;
                    for col in 0..self.out_dim {
                        out_row[col] += bias_data[bias_offset + col];
                    }
                });
        }

        self.cache.log_stats();

        let result =
            Tensor::from_vec(output, (num_tokens, topk, self.out_dim), &Device::Cpu)?;
        let result = result.to_device(x.device())?.to_dtype(x.dtype())?;
        Ok(result)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::BF16, Device::Cpu)
    }

    fn plan_isq(&self, _request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        candle_core::bail!("{} does not support ISQ", self.name())
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("{} does not support add_delta_w", self.name())
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("{} does not support ISQ", self.name())
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

impl QuantizedSerde for MxFp4StreamingExpertLayer {
    fn name(&self) -> &'static str {
        "mxfp4-expert-stream"
    }
}

impl MXFP4Layer {
    pub(crate) fn inspect_uqff_header(layer: &UqffLayerHeaderView<'_>) -> Option<UqffHeaderMatch> {
        const WEIGHT_SUFFIXES: &[&str] = &["weight", "weight.format", "weight.scales"];
        if layer.exact_weight_suffixes(WEIGHT_SUFFIXES) && layer.scalar("weight.format", Dtype::U8)
        {
            Some(UqffHeaderMatch {
                serde_type: QuantizedSerdeType::Mxfp4,
            })
        } else {
            None
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        _tensors: &[UqffTensor],
        _prefix: &str,
    ) -> Result<String> {
        Ok("mxfp4".to_string())
    }
}

impl QuantMethod for MXFP4Layer {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Afq { .. } => unreachable!(),
            QuantMethodConfig::MXFP4 {
                blocks,
                scales,
                bias,
            } => Ok(Self {
                blocks,
                scales,
                bias,
            }),
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        self.dequantize_weights_to(self.blocks.device())
    }

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, k_half) = self.blocks.dims2()?;
        let mut output_shape = ids.dims().to_vec();
        output_shape.push(k_half * 2);

        let ids = ids
            .to_device(self.blocks.device())?
            .flatten_all()?
            .contiguous()?;
        let blocks = self.blocks.index_select(&ids, 0)?;
        let scales = self.scales.index_select(&ids, 0)?;
        Self::dequantize_rows(&blocks, &scales)?.reshape(output_shape)
    }

    #[allow(unused_variables)]
    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(x.device(), Device::Cuda(_)) && ffi::HAVE_MXFP4_GEMM_KERNELS {
            let orig_dims = x.dims().to_vec();
            let x_2d = if orig_dims.len() > 2 {
                let features = orig_dims[orig_dims.len() - 1];
                let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                x.reshape((batch_size, features))?
            } else {
                x.clone()
            };

            let result = ops::mxfp4_matmul(&x_2d, &self.blocks, &self.scales, self.bias.as_ref())?;

            if orig_dims.len() > 2 {
                let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                new_dims.push(result.dim(1)?);
                return result.reshape(new_dims);
            }
            return Ok(result);
        }

        #[cfg(feature = "metal")]
        {
            if x.device().is_metal() {
                let orig_dims = x.dims().to_vec();
                let x_2d = if orig_dims.len() > 2 {
                    let features = orig_dims[orig_dims.len() - 1];
                    let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                    x.reshape((batch_size, features))?
                } else {
                    x.clone()
                };

                let result =
                    metal_ops::mxfp4_matmul(&x_2d, &self.blocks, &self.scales, self.bias.as_ref())?;

                if orig_dims.len() > 2 {
                    let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                    new_dims.push(result.dim(1)?);
                    return result.reshape(new_dims);
                }
                return Ok(result);
            }
        }

        self.forward_dequantize(x)
    }

    #[allow(unused_variables)]
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(x.device(), Device::Cuda(_)) && ffi::HAVE_MXFP4_GEMM_KERNELS {
            return ops::mxfp4_indexed_moe_gemm(
                x,
                &self.blocks,
                &self.scales,
                self.bias.as_ref(),
                indices,
            );
        }

        #[cfg(feature = "metal")]
        {
            if x.device().is_metal() {
                return metal_ops::mxfp4_indexed_moe_gemm(
                    x,
                    &self.blocks,
                    &self.scales,
                    self.bias.as_ref(),
                    indices,
                );
            }
        }

        self.gather_forward_dequantize(x, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("MXFP4Layer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::BF16, self.scales.device().clone())
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        let mut shape = self.blocks.dims().to_vec();
        if let Some(last) = shape.last_mut() {
            *last = last.saturating_mul(2);
        }
        if shape.len() == 3 && request.ty.is_some_and(|ty| !Self::supports_stacked_isq(ty)) {
            candle_core::bail!(
                "Cannot requantize packed MXFP4 expert weights to {}: that target does not support stacked expert gather. Use a Q*K/Q*_0/Q*_1 target, AFQ, MXFP4, or omit ISQ.",
                request.ty.expect("rank-3 rejection requires an ISQ target")
            );
        }
        Ok(crate::plan_weight_isq(
            DType::BF16,
            self.scales.device().clone(),
            shape,
            request,
            true,
        ))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() || (dtype == Some(IsqType::MXFP4) && imatrix_weight.is_none()) {
            let blocks = self.blocks.to_device(&device)?;
            let scales = self.scales.to_device(&device)?;
            let bias = self
                .bias
                .as_ref()
                .map(|bias| bias.to_device(&device))
                .transpose()?;
            return Ok(Arc::new(Self::from_parts(blocks, scales, bias)));
        }

        let weight = self.dequantize_weights_to(&Device::Cpu)?;
        let bias = self
            .bias
            .as_ref()
            .map(|bias| bias.to_device(&Device::Cpu))
            .transpose()?;

        if weight.rank() == 3 {
            let Some(dtype) = dtype else {
                return Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                    candle_nn::Linear::new(weight, bias),
                ))?)
                .apply_isq(None, device, n_quantized, imatrix_weight, guard);
            };

            if candle_core::quantized::GgmlDType::try_from(dtype).is_ok() {
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let weight = crate::GgufMatMul::quantize_expert_stack(
                    &weight,
                    dtype,
                    imatrix_weight.as_deref(),
                    &device,
                    guard,
                )?;
                let bias = bias
                    .map(|bias| bias.to_dtype(DType::F32)?.to_device(&device))
                    .transpose()?;
                return Ok(Arc::new(crate::GgufMatMul::from_qtensor(weight, bias)));
            }

            if !Self::supports_stacked_isq(dtype) {
                candle_core::bail!(
                    "Cannot requantize packed MXFP4 expert weights to {dtype}: that target does not support stacked expert gather. Use a Q*K/Q*_0/Q*_1 target, AFQ, MXFP4, or omit ISQ."
                );
            }
        }

        Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
            candle_nn::Linear::new(weight, bias),
        ))?)
        .apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }
}

impl MXFP4Layer {
    pub fn supports_stacked_isq(ty: IsqType) -> bool {
        ty.supports_stacked_gather() || ty == IsqType::MXFP4
    }

    pub fn from_parts(blocks: Tensor, scales: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            blocks,
            scales,
            bias,
        }
    }

    fn from_uqff(reader: &UqffReader, key: &str, device: &Device, shard: Shard) -> Result<Self> {
        // Logical dims: blocks pack 2 FP4 input elements per byte along the last dim.
        let blocks_dims = reader.tensor_dims(&format!("{key}.weight"))?;
        let mut dims = blocks_dims.clone();
        *dims.last_mut().expect("MXFP4 blocks are non-empty") *= 2;
        let range = crate::uqff::shard_range(shard, &dims)?;
        let (blocks_range, scales_range) = match range {
            None => (None, None),
            Some((dim, start, len)) if dim == dims.len() - 1 => {
                if !start.is_multiple_of(MXFP4_BLOCK_SIZE) || !len.is_multiple_of(MXFP4_BLOCK_SIZE)
                {
                    candle_core::bail!(
                        "Sharding the MXFP4 packed dim requires alignment of {MXFP4_BLOCK_SIZE}: start {start}, len {len}."
                    );
                }
                (
                    Some((dim, start / 2, len / 2)),
                    Some((dim, start / MXFP4_BLOCK_SIZE, len / MXFP4_BLOCK_SIZE)),
                )
            }
            some => (some, some),
        };
        let blocks = reader.load_tensor_sharded(&format!("{key}.weight"), device, blocks_range)?;
        let scales =
            reader.load_tensor_sharded(&format!("{key}.weight.scales"), device, scales_range)?;
        let bias = reader.load_bias(key, device, range, dims.len())?;
        Ok(Self::from_parts(blocks, scales, bias))
    }

    /// Check if the device supports MXFP4 operations.
    ///
    /// CPU support uses the blockwise dequantize + matmul fallback implemented
    /// in forward_dequantize / gather_forward_dequantize.
    fn device_supported(_device: &Device) -> bool {
        if _device.is_cpu() {
            return true;
        }
        #[cfg(feature = "cuda")]
        if matches!(_device, Device::Cuda(_)) {
            return ffi::HAVE_MXFP4_GEMM_KERNELS;
        }
        #[cfg(feature = "metal")]
        if _device.is_metal() {
            return true;
        }
        false
    }

    /// Construct one logical [rows, cols] MXFP4 matrix from canonical GGUF bytes.
    ///
    /// GGUF dtype 39 stores each 32-value block as one E8M0 scale byte followed by 16 packed
    /// FP4 bytes in GGML's split-half layout: byte j low nibble is element j and high nibble
    /// is element j + 16. Candle's packed representation pairs adjacent elements in each byte,
    /// so the raw payload must be losslessly repacked.
    pub fn from_gguf_bytes(
        data: &[u8],
        rows: usize,
        cols: usize,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        if cols == 0 || !cols.is_multiple_of(MXFP4_BLOCK_SIZE) {
            candle_core::bail!(
                "MXFP4 streamed matrix requires cols divisible by {}, got {}",
                MXFP4_BLOCK_SIZE,
                cols
            );
        }

        let blocks_per_row = cols / MXFP4_BLOCK_SIZE;
        let packed_bytes_per_block = MXFP4_BLOCK_SIZE / 2;
        let row_bytes = blocks_per_row * (1 + packed_bytes_per_block);
        let expected = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| candle_core::Error::Msg("MXFP4 byte-size overflow".to_string()))?;
        if data.len() != expected {
            candle_core::bail!(
                "MXFP4 streamed matrix has {} bytes, expected {}",
                data.len(),
                expected
            );
        }

        let mut blocks = Vec::with_capacity(rows * cols / 2);
        let mut scales = Vec::with_capacity(rows * blocks_per_row);

        for row in 0..rows {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            for block in 0..blocks_per_row {
                let start = block * (1 + packed_bytes_per_block);
                scales.push(row_data[start]);
                let native = &row_data[start + 1..start + 1 + packed_bytes_per_block];

                // GGML MXFP4 is split-half:
                //   native[j].lo -> element j
                //   native[j].hi -> element j + 16.
                // Repack to Candle's adjacent-pair layout.
                for pair in 0..packed_bytes_per_block {
                    let elem0 = pair * 2;
                    let elem1 = elem0 + 1;
                    let nibble = |element: usize| -> u8 {
                        if element < 16 {
                            native[element] & 0x0f
                        } else {
                            native[element - 16] >> 4
                        }
                    };
                    blocks.push(nibble(elem0) | (nibble(elem1) << 4));
                }
            }
        }

        let blocks = Tensor::from_vec(blocks, (rows, cols / 2), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let scales = Tensor::from_vec(scales, (rows, blocks_per_row), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;

        Ok(Self::from_parts(blocks, scales, bias))
    }

    /// Quantize an unquantized weight tensor to MXFP4 format.
    /// weight shape: `[N, K]`, bias shape: `[N]` (optional)
    pub fn quantize(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight_f32 = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let dims = weight_f32.dims2()?;
        let (n, k) = (dims.0, dims.1);

        if k % MXFP4_BLOCK_SIZE != 0 {
            candle_core::bail!(
                "MXFP4 quantization requires K ({k}) divisible by block size ({MXFP4_BLOCK_SIZE})"
            );
        }

        let weight_data: Vec<f32> = weight_f32.flatten_all()?.to_vec1()?;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let k_half = k / 2;

        // Parallelize quantization across rows with rayon
        use rayon::prelude::*;
        let row_results: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let row_offset = row * k;
                let mut row_packed = vec![0u8; k_half];
                let mut row_scales = vec![0u8; num_blocks_per_row];

                for (blk, row_scale) in row_scales.iter_mut().enumerate() {
                    let blk_start = row_offset + blk * MXFP4_BLOCK_SIZE;
                    let block = &weight_data[blk_start..blk_start + MXFP4_BLOCK_SIZE];

                    let max_abs = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

                    let scale = if max_abs == 0.0 {
                        127u8
                    } else {
                        let raw = (max_abs / 6.0).log2().floor() as i32 + 127;
                        raw.clamp(0, 254) as u8
                    };
                    *row_scale = scale;

                    let scale_factor = 2.0f32.powi(scale as i32 - 127);
                    let inv_scale = if scale_factor == 0.0 {
                        0.0
                    } else {
                        1.0 / scale_factor
                    };

                    for (elem, &val) in block.iter().enumerate() {
                        let nibble = Self::quantize_to_fp4(val * inv_scale);
                        let k_idx = blk * MXFP4_BLOCK_SIZE + elem;
                        let byte_idx = k_idx / 2;
                        if k_idx.is_multiple_of(2) {
                            row_packed[byte_idx] |= nibble;
                        } else {
                            row_packed[byte_idx] |= nibble << 4;
                        }
                    }
                }
                (row_packed, row_scales)
            })
            .collect();

        let mut packed = Vec::with_capacity(n * k_half);
        let mut scales = Vec::with_capacity(n * num_blocks_per_row);
        for (row_packed, row_scales) in row_results {
            packed.extend_from_slice(&row_packed);
            scales.extend_from_slice(&row_scales);
        }

        let blocks = Tensor::from_vec(packed, (n, k / 2), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let scales = Tensor::from_vec(scales, (n, num_blocks_per_row), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let bias = bias.map(|b| b.to_device(device)).transpose()?;

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Quantize a single scaled value to the nearest FP4 E2M1 nibble (0..15).
    fn quantize_to_fp4(val: f32) -> u8 {
        // FP4 E2M1 positive values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        // Negative values are the same with sign bit set (indices 8..15)
        let sign = val < 0.0;
        let abs_val = val.abs();

        // Decision boundaries (midpoints between consecutive FP4 values)
        let nibble = if abs_val < 0.25 {
            0 // 0.0
        } else if abs_val < 0.75 {
            1 // 0.5
        } else if abs_val < 1.25 {
            2 // 1.0
        } else if abs_val < 1.75 {
            3 // 1.5
        } else if abs_val < 2.5 {
            4 // 2.0
        } else if abs_val < 3.5 {
            5 // 3.0
        } else if abs_val < 5.0 {
            6 // 4.0
        } else {
            7 // 6.0
        };

        if sign {
            nibble | 0x08
        } else {
            nibble
        }
    }

    pub fn linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !Self::device_supported(vb.device()) {
            candle_core::bail!("MXFP4Layer requires a supported CPU, CUDA, or Metal device.");
        }

        let QuantizedConfig::MXFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let blocks = vb.get_with_hints_dtype(
            (out_dim, in_dim / 2),
            "blocks",
            Default::default(),
            DType::U8,
        )?;
        let scales = vb.get_with_hints_dtype(
            (out_dim, in_dim / MXFP4_BLOCK_SIZE),
            "scales",
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    pub fn packed_linear_b(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !Self::device_supported(vb.device()) {
            candle_core::bail!("MXFP4Layer requires CUDA or Metal device.");
        }

        let QuantizedConfig::MXFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let blocks = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim / 2),
            "blocks",
            Default::default(),
            DType::U8,
        )?;
        let scales = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim / MXFP4_BLOCK_SIZE),
            "scales",
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Load GPT-OSS style MXFP4 experts (combined gate_up_proj format).
    ///
    /// GPT-OSS stores tensors as:
    /// - `{name}_blocks`: [num_experts, out_dim, num_blocks, 16] where 16 bytes = 32 FP4 values
    /// - `{name}_scales`: [num_experts, out_dim, num_blocks]
    /// - `{name}_bias`: [num_experts, out_dim]
    ///
    /// This function loads and reshapes the 4D blocks tensor to 3D [num_experts, out_dim, in_dim/2].
    pub fn packed_gptoss_linear(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        name: &str,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let num_blocks = in_dim / MXFP4_BLOCK_SIZE;

        let blocks_4d = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, num_blocks, 16),
            &format!("{name}_blocks"),
            Default::default(),
            DType::U8,
        )?;

        let blocks = blocks_4d.reshape((num_local_experts, out_dim, num_blocks * 16))?;

        let scales = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, num_blocks),
            &format!("{name}_scales"),
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), &format!("{name}_bias"))?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Combined FP4 × E8M0 dequant table: `DEQUANT_LUT[scale][nibble]`.
    /// For each of the 256 possible E8M0 scale values, stores the 16 possible
    /// dequantized values (FP4_LUT[nibble] * 2^(scale - 127)).
    /// This turns dequantization into a single table lookup per element.
    const DEQUANT_LUT: [[f32; 16]; 256] = {
        let mut lut = [[0.0f32; 16]; 256];
        let fp4: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut s = 0u32;
        while s < 256 {
            // GGML E8M0 uses the special encodings x=0,1 for the subnormal
            // floor, then regular IEEE-754 exponents for x>=2.
            let scale_factor = if s < 2 {
                f32::from_bits(0x0020_0000u32 << s)
            } else {
                f32::from_bits((s - 1) << 23)
            };
            let mut n = 0;
            while n < 16 {
                lut[s as usize][n] = fp4[n] * scale_factor;
                n += 1;
            }
            s += 1;
        }
        lut
    };

    /// Dequantize MXFP4 weights to f32
    /// blocks: [num_experts, N, K/2] packed bytes
    /// scales: [num_experts, N, K/32] E8M0 scales
    /// Returns: [num_experts, N, K] f32 weights
    fn dequantize_weights_to(&self, device: &Device) -> Result<Tensor> {
        let blocks_dims = self.blocks.dims();

        let (num_experts, n, k_half) = if blocks_dims.len() == 3 {
            (blocks_dims[0], blocks_dims[1], blocks_dims[2])
        } else {
            (1, blocks_dims[0], blocks_dims[1])
        };
        let k = k_half * 2;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;

        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;

        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;

        let mut weights = vec![0f32; num_experts * n * k];
        let half_block = MXFP4_BLOCK_SIZE / 2; // 16 packed bytes per block

        for expert in 0..num_experts {
            for row in 0..n {
                let blocks_row = expert * n * k_half + row * k_half;
                let scales_row = expert * n * num_blocks_per_row + row * num_blocks_per_row;
                let weights_row = expert * n * k + row * k;

                for blk in 0..num_blocks_per_row {
                    let scale = scales_data[scales_row + blk] as usize;
                    let dequant = &Self::DEQUANT_LUT[scale];
                    let blk_bytes = &blocks_data[blocks_row + blk * half_block..];
                    let w_out = &mut weights[weights_row + blk * MXFP4_BLOCK_SIZE..];

                    for byte_i in 0..half_block {
                        let packed = blk_bytes[byte_i];
                        w_out[byte_i * 2] = dequant[(packed & 0x0F) as usize];
                        w_out[byte_i * 2 + 1] = dequant[((packed >> 4) & 0x0F) as usize];
                    }
                }
            }
        }

        let shape = if blocks_dims.len() == 3 {
            vec![num_experts, n, k]
        } else {
            vec![n, k]
        };

        Tensor::from_vec(weights, shape.as_slice(), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(device)
    }

    fn dequantize_rows(blocks: &Tensor, scales: &Tensor) -> Result<Tensor> {
        use rayon::prelude::*;

        let (num_rows, k_half) = blocks.dims2()?;
        let k = k_half * 2;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let half_block = MXFP4_BLOCK_SIZE / 2;
        let blocks_data = blocks
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let scales_data = scales
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let mut weights = vec![0f32; num_rows * k];

        weights
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(row, weights_row)| {
                let blocks_row = row * k_half;
                let scales_row = row * num_blocks_per_row;
                for blk in 0..num_blocks_per_row {
                    let scale = scales_data[scales_row + blk] as usize;
                    let dequant = &Self::DEQUANT_LUT[scale];
                    let block_start = blocks_row + blk * half_block;
                    let output_start = blk * MXFP4_BLOCK_SIZE;
                    for byte_i in 0..half_block {
                        let packed = blocks_data[block_start + byte_i];
                        weights_row[output_start + byte_i * 2] = dequant[(packed & 0x0F) as usize];
                        weights_row[output_start + byte_i * 2 + 1] =
                            dequant[((packed >> 4) & 0x0F) as usize];
                    }
                }
            });

        Tensor::from_vec(weights, (num_rows, k), &Device::Cpu)?
            .to_device(blocks.device())?
            .to_dtype(DType::BF16)
    }

    /// CPU forward pass: zero-copy packed-weight access + blocked dequant/matmul.
    /// Processes MXFP4_BLOCK_SIZE (32) input columns at a time, dequantizing only
    /// the current weight block before accumulating partial results.
    fn forward_dequantize(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();

        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        let x_f32 = x_2d.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (m, k) = x_f32.dims2()?;
        if !k.is_multiple_of(MXFP4_BLOCK_SIZE) {
            candle_core::bail!(
                "MXFP4 CPU fallback requires K ({k}) divisible by block size ({MXFP4_BLOCK_SIZE})"
            );
        }

        let blocks_dims = self.blocks.dims();
        let (n, weight_k_half) = match blocks_dims {
            [n, k_half] => (*n, *k_half),
            _ => candle_core::bail!(
                "MXFP4 CPU forward expects rank-2 packed weights, got rank {}",
                blocks_dims.len()
            ),
        };
        let expected_k = weight_k_half * 2;
        if k != expected_k {
            candle_core::bail!(
                "MXFP4 CPU fallback input K {k} does not match packed weight K {expected_k}"
            );
        }

        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let half_block = MXFP4_BLOCK_SIZE / 2;

        let (blocks_storage, blocks_layout) = self.blocks.storage_and_layout();
        let (scales_storage, scales_layout) = self.scales.storage_and_layout();
        let (blocks_start, blocks_end) = blocks_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg(
                "MXFP4 CPU fallback requires contiguous packed weights".into(),
            ))?;
        let (scales_start, scales_end) = scales_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg(
                "MXFP4 CPU fallback requires contiguous scales".into(),
            ))?;
        let blocks_data = match &*blocks_storage {
            Storage::Cpu(storage) => &storage.as_slice::<u8>()?[blocks_start..blocks_end],
            _ => candle_core::bail!("MXFP4 CPU fallback requires CPU packed weights"),
        };
        let scales_data = match &*scales_storage {
            Storage::Cpu(storage) => &storage.as_slice::<u8>()?[scales_start..scales_end],
            _ => candle_core::bail!("MXFP4 CPU fallback requires CPU scales"),
        };
        if scales_data.len() != n * num_blocks_per_row {
            candle_core::bail!(
                "MXFP4 CPU fallback scales shape does not match packed weights"
            );
        }
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;

        // output: [m, n], accumulate x @ W^T in blocks of 32 columns
        let mut output = vec![0f32; m * n];
        let k_half = k / 2;

        for blk in 0..num_blocks_per_row {
            let col_start = blk * MXFP4_BLOCK_SIZE;

            for row in 0..n {
                let scale = scales_data[row * num_blocks_per_row + blk] as usize;
                let dequant = &Self::DEQUANT_LUT[scale];
                let blk_bytes = &blocks_data[row * k_half + blk * half_block..];

                // Dequantize this block of 32 weights for this output row
                let mut w_block = [0f32; MXFP4_BLOCK_SIZE];
                for byte_i in 0..half_block {
                    let packed = blk_bytes[byte_i];
                    w_block[byte_i * 2] = dequant[(packed & 0x0F) as usize];
                    w_block[byte_i * 2 + 1] = dequant[((packed >> 4) & 0x0F) as usize];
                }

                // Accumulate dot product for all tokens against this weight block
                for token in 0..m {
                    let x_row = &x_data[token * k + col_start..];
                    let mut acc = 0f32;
                    for i in 0..MXFP4_BLOCK_SIZE {
                        acc += x_row[i] * w_block[i];
                    }
                    output[token * n + row] += acc;
                }
            }
        }

        let mut result = Tensor::from_vec(output, (m, n), &Device::Cpu)?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;

        if let Some(bias) = &self.bias {
            result = result.broadcast_add(bias)?;
        }

        if orig_dims.len() > 2 {
            let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
            new_dims.push(result.dim(1)?);
            result = result.reshape(new_dims)?;
        }

        Ok(result)
    }

    /// CPU MoE forward: zero-copy packed-weight access and blocked dequant per
    /// (token, expert) pair. Only the selected expert blocks are traversed.
    fn gather_forward_dequantize(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims();
        let indices_dims = indices.dims();
        if indices_dims.len() != 2 {
            candle_core::bail!(
                "MXFP4 CPU MoE fallback expects rank-2 expert indices, got rank {}",
                indices_dims.len()
            );
        }

        let (num_tokens, topk, k, x_has_topk) = if x_dims.len() == 2 {
            (x_dims[0], indices_dims[1], x_dims[1], false)
        } else if x_dims.len() == 3 {
            if x_dims[0] != indices_dims[0] || x_dims[1] != indices_dims[1] {
                candle_core::bail!("MXFP4 CPU MoE input and indices shapes do not agree");
            }
            (x_dims[0], x_dims[1], x_dims[2], true)
        } else {
            candle_core::bail!(
                "MXFP4 CPU MoE fallback expects rank-2 or rank-3 input, got rank {}",
                x_dims.len()
            );
        };

        if !k.is_multiple_of(MXFP4_BLOCK_SIZE) {
            candle_core::bail!(
                "MXFP4 CPU MoE fallback requires K ({k}) divisible by block size ({MXFP4_BLOCK_SIZE})"
            );
        }

        let blocks_dims = self.blocks.dims();
        let (n, weight_k_half) = if blocks_dims.len() == 3 {
            (blocks_dims[1], blocks_dims[2])
        } else {
            candle_core::bail!("MXFP4 CPU MoE fallback expects rank-3 packed weights");
        };
        let expected_k = weight_k_half * 2;
        if k != expected_k {
            candle_core::bail!(
                "MXFP4 CPU MoE fallback input K {k} does not match packed weight K {expected_k}"
            );
        }

        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let k_half = k / 2;
        let half_block = MXFP4_BLOCK_SIZE / 2;

        let (blocks_storage, blocks_layout) = self.blocks.storage_and_layout();
        let (scales_storage, scales_layout) = self.scales.storage_and_layout();
        let (blocks_start, blocks_end) = blocks_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg(
                "MXFP4 CPU MoE fallback requires contiguous packed weights".into(),
            ))?;
        let (scales_start, scales_end) = scales_layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg(
                "MXFP4 CPU MoE fallback requires contiguous scales".into(),
            ))?;
        let blocks_data = match &*blocks_storage {
            Storage::Cpu(storage) => &storage.as_slice::<u8>()?[blocks_start..blocks_end],
            _ => candle_core::bail!("MXFP4 CPU MoE fallback requires CPU packed weights"),
        };
        let scales_data = match &*scales_storage {
            Storage::Cpu(storage) => &storage.as_slice::<u8>()?[scales_start..scales_end],
            _ => candle_core::bail!("MXFP4 CPU MoE fallback requires CPU scales"),
        };
        if scales_data.len() != self.blocks.dims()[0] * n * num_blocks_per_row {
            candle_core::bail!(
                "MXFP4 CPU MoE fallback scales shape does not match packed weights"
            );
        }

        let x_f32 = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;

        let indices_cpu = indices.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;
        let indices_data: Vec<u32> = indices_cpu.flatten_all()?.to_vec1()?;

        let bias_data: Option<Vec<f32>> = self
            .bias
            .as_ref()
            .map(|b| {
                b.to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1()
            })
            .transpose()?;

        if let Some(bias) = &bias_data {
            let expected = self.blocks.dims()[0] * n;
            if bias.len() != expected {
                candle_core::bail!(
                    "MXFP4 CPU MoE bias has {} elements, expected {}",
                    bias.len(),
                    expected
                );
            }
        }

        let num_experts = self.blocks.dims()[0];

        // output: [num_tokens * topk, n]
        let mut output = vec![0f32; num_tokens * topk * n];

        for token_idx in 0..num_tokens {
            for slot_idx in 0..topk {
                let expert_idx = indices_data[token_idx * topk + slot_idx] as usize;
                if expert_idx >= num_experts {
                    candle_core::bail!(
                        "MXFP4 CPU MoE expert index {} out of range for {} experts",
                        expert_idx,
                        num_experts
                    );
                }
                let out_row = token_idx * topk + slot_idx;

                // Get input row
                let x_offset = if x_has_topk {
                    (token_idx * topk + slot_idx) * k
                } else {
                    token_idx * k
                };

                // Blocked dequant + matmul for this (token, expert) pair
                let expert_blocks_base = expert_idx * n * k_half;
                let expert_scales_base = expert_idx * n * num_blocks_per_row;

                for blk in 0..num_blocks_per_row {
                    let col_start = blk * MXFP4_BLOCK_SIZE;

                    // Load input block
                    let x_blk =
                        &x_data[x_offset + col_start..x_offset + col_start + MXFP4_BLOCK_SIZE];

                    for row in 0..n {
                        let scale = scales_data[expert_scales_base + row * num_blocks_per_row + blk]
                            as usize;
                        let dequant = &Self::DEQUANT_LUT[scale];
                        let blk_bytes =
                            &blocks_data[expert_blocks_base + row * k_half + blk * half_block..];

                        let mut dot = 0f32;
                        for byte_i in 0..half_block {
                            let packed = blk_bytes[byte_i];
                            let w0 = dequant[(packed & 0x0F) as usize];
                            let w1 = dequant[((packed >> 4) & 0x0F) as usize];
                            dot += x_blk[byte_i * 2] * w0 + x_blk[byte_i * 2 + 1] * w1;
                        }
                        output[out_row * n + row] += dot;
                    }
                }

                // Add bias
                if let Some(ref bias) = bias_data {
                    let bias_offset = expert_idx * n;
                    for row in 0..n {
                        output[out_row * n + row] += bias[bias_offset + row];
                    }
                }
            }
        }

        let result = Tensor::from_vec(output, (num_tokens * topk, n), &Device::Cpu)?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        result.reshape((num_tokens, topk, n))
    }
}

impl QuantizedSerde for MXFP4Layer {
    fn name(&self) -> &'static str {
        "mxfp4-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn uqff_type(&self) -> Option<IsqType> {
        Some(IsqType::MXFP4)
    }
    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        if ty != IsqType::MXFP4 {
            candle_core::bail!("Cannot serialize MXFP4 layer as {ty}; actual type is MXFP4.");
        }

        let mut data = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Mxfp4 as u8,
            ),
            UqffTensor::from_tensor(format!("{prefix}.weight"), &self.blocks)?,
            UqffTensor::from_tensor(format!("{prefix}.weight.scales"), &self.scales)?,
        ];
        if let Some(bias) = &self.bias {
            data.push(UqffTensor::from_tensor(format!("{prefix}.bias"), bias)?);
        }
        Ok(data)
    }
    fn deserialize_uqff(
        reader: &UqffReader,
        prefix: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self::from_uqff(reader, prefix, device, shard)?))
    }
    fn isq_type_from_uqff(_reader: &UqffReader, _prefix: &str) -> Result<IsqType> {
        Ok(IsqType::MXFP4)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpu_device_support_is_enabled() {
        assert!(MXFP4Layer::device_supported(&Device::Cpu));
    }

    #[test]
    fn streaming_dot_block_kernel_matches_scalar() -> Result<()> {
        let x = (0..MXFP4_BLOCK_SIZE)
            .map(|i| (i as f32 * 0.03125) - 0.5)
            .collect::<Vec<_>>();
        let w = (0..MXFP4_BLOCK_SIZE)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.125)
            .collect::<Vec<_>>();
        let scalar = MXFP4_BLOCK_SIZE
            .checked_sub(0)
            .map(|_| x.iter().zip(&w).map(|(a, b)| a * b).sum::<f32>())
            .unwrap();
        let kernel = MxFp4StreamingExpertLayer::dot_kernel();
        let actual = MxFp4StreamingExpertLayer::dot_block(&x, &w, kernel);
        assert!((actual - scalar).abs() < 1e-4, "actual={actual} scalar={scalar}");
        Ok(())
    }

    #[test]
    fn streaming_dot_row_uses_output_column_once() -> Result<()> {
        let x = vec![1.0f32; MXFP4_BLOCK_SIZE];
        // Four output rows, one 32-value MXFP4 block per row: 1 scale byte + 16 packed bytes.
        let raw_expert = vec![127u8; 4 * (MXFP4_BLOCK_SIZE / 2 + 1)];
        let mut output = vec![0.0f32; 4];

        // This targets the last output column. The helper now receives the
        // route-local output row, so the column offset is applied exactly once.
        MxFp4StreamingExpertLayer::dot_row(
            &x,
            &raw_expert,
            3,
            &mut output,
            3,
        );

        assert!(output[3].is_finite());
        Ok(())
    }


    use super::*;

    const TEST_HIDDEN_SIZE: usize = 64;
    const TEST_VOCAB_SIZE: usize = 7;

    fn test_layer() -> Result<Arc<dyn QuantMethod>> {
        let values = (0..TEST_VOCAB_SIZE * TEST_HIDDEN_SIZE)
            .map(|index| ((index % 29) as f32 - 14.0) / 3.0)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(values, (TEST_VOCAB_SIZE, TEST_HIDDEN_SIZE), &Device::Cpu)?;
        MXFP4Layer::quantize(&weight, None, &Device::Cpu)
    }

    fn expected_embedding(layer: &dyn QuantMethod, ids: &Tensor) -> Result<Tensor> {
        let weight = layer.dequantize_w()?;
        let mut shape = ids.dims().to_vec();
        shape.push(TEST_HIDDEN_SIZE);
        let ids = ids
            .to_device(weight.device())?
            .flatten_all()?
            .contiguous()?;
        weight.index_select(&ids, 0)?.reshape(shape)
    }

    fn stacked_test_layer() -> Result<Arc<MXFP4Layer>> {
        const EXPERTS: usize = 2;
        const OUTPUT: usize = 4;
        let blocks = Tensor::from_vec(
            vec![0x22u8; EXPERTS * OUTPUT * TEST_HIDDEN_SIZE / 2],
            (EXPERTS, OUTPUT, TEST_HIDDEN_SIZE / 2),
            &Device::Cpu,
        )?;
        let scales = Tensor::from_vec(
            vec![127u8; EXPERTS * OUTPUT * TEST_HIDDEN_SIZE / MXFP4_BLOCK_SIZE],
            (EXPERTS, OUTPUT, TEST_HIDDEN_SIZE / MXFP4_BLOCK_SIZE),
            &Device::Cpu,
        )?;
        let bias = Tensor::from_vec(
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            (EXPERTS, OUTPUT),
            &Device::Cpu,
        )?;
        Ok(Arc::new(MXFP4Layer::from_parts(blocks, scales, Some(bias))))
    }

    fn stacked_test_inputs() -> Result<(Tensor, Tensor)> {
        let input = Tensor::ones((2, 1, TEST_HIDDEN_SIZE), DType::F32, &Device::Cpu)?;
        let indices = Tensor::from_vec(vec![0u32, 1], (2, 1), &Device::Cpu)?;
        Ok((input, indices))
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn cpu_forward_matches_dequantized_matmul() -> Result<()> {
        let layer = test_layer()?;
        let input = Tensor::from_vec(
            (0..2 * TEST_HIDDEN_SIZE)
                .map(|i| ((i % 11) as f32 - 5.0) / 7.0)
                .collect::<Vec<_>>(),
            (2, TEST_HIDDEN_SIZE),
            &Device::Cpu,
        )?;

        let actual = layer.forward(&input)?;
        let weights = layer.dequantize_w()?.to_dtype(DType::F32)?;
        let expected = input
            .matmul(&weights.t()?)?
            .to_dtype(DType::F32)?;

        assert_eq!(actual.dims(), &[2, TEST_VOCAB_SIZE]);
        assert_close(&actual, &expected, 1e-3)
    }

    #[test]
    fn embedding_selects_quantized_rows() -> Result<()> {
        let layer = test_layer()?;
        let ids = Tensor::from_vec(vec![6u32, 1, 6, 3, 0, 4], (2, 3), &Device::Cpu)?;

        let output = layer.embedding_forward(&ids, DType::F32)?;
        let expected = expected_embedding(layer.as_ref(), &ids)?.to_dtype(DType::F32)?;

        assert_eq!(output.dims(), &[2, 3, TEST_HIDDEN_SIZE]);
        assert_eq!(output.dtype(), DType::F32);
        assert!(output.device().is_cpu());
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn uqff_loaded_embedding_selects_quantized_rows() -> Result<()> {
        let layer = test_layer()?;
        let mut tensors = crate::uqff_version_tensors();
        tensors.extend(layer.serialize_uqff("test.embedding", IsqType::MXFP4)?);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mistralrs-mxfp4-embedding-uqff-{}-{stamp}.uqff",
            std::process::id()
        ));
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .map_err(candle_core::Error::wrap)?;

        let reader = UqffReader::open(std::slice::from_ref(&path))?;
        let loaded = reader
            .load_linear("test.embedding", &Device::Cpu, Shard::default())?
            .unwrap();
        let ids = Tensor::from_vec(vec![5u32, 2, 1, 5], (2, 2), &Device::Cpu)?;
        let output = loaded.embedding_forward(&ids, DType::BF16)?;
        let expected = expected_embedding(layer.as_ref(), &ids)?;
        drop(reader);
        let _ = std::fs::remove_file(path);

        assert_eq!(output.dims(), &[2, 2, TEST_HIDDEN_SIZE]);
        assert_eq!(output.dtype(), DType::BF16);
        assert!(output.device().is_cpu());
        assert_eq!(
            output.flatten_all()?.to_vec1::<half::bf16>()?,
            expected.flatten_all()?.to_vec1::<half::bf16>()?
        );
        Ok(())
    }

    #[test]
    fn gguf_mxfp4_bytes_repack_split_half_layout() -> Result<()> {
        let mut block = vec![127u8];
        for j in 0..(MXFP4_BLOCK_SIZE / 2) {
            // element j uses the low nibble; element j+16 uses the high nibble.
            let lo = j as u8;
            let hi = (15 - j) as u8;
            block.push(lo | (hi << 4));
        }

        let layer = MXFP4Layer::from_gguf_bytes(
            &block,
            1,
            MXFP4_BLOCK_SIZE,
            None,
            &Device::Cpu,
        )?;
        let actual = layer.dequantize_w()?.to_dtype(DType::F32)?;
        let fp4 = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let expected_values = (0..16)
            .map(|j| fp4[j])
            .chain((0..16).map(|j| fp4[15 - j]))
            .collect::<Vec<_>>();
        let expected = Tensor::from_vec(expected_values, (1, MXFP4_BLOCK_SIZE), &Device::Cpu)?;
        assert_close(&actual, &expected)?;
        Ok(())
    }

    #[test]
    fn mxfp4_e8m0_endpoints_match_ggml() -> Result<()> {
        assert_eq!(MXFP4Layer::DEQUANT_LUT[0][2], 2.0f32.powi(-126));
        assert_eq!(MXFP4Layer::DEQUANT_LUT[1][2], 2.0f32.powi(-125));
        assert_eq!(MXFP4Layer::DEQUANT_LUT[255][2], 2.0f32.powi(127));
        Ok(())
    }

    #[test]
    fn stacked_mxfp4_requantizes_to_ggml_with_bias() -> Result<()> {
        let layer = stacked_test_layer()?;
        let (input, indices) = stacked_test_inputs()?;
        let expected = layer.gather_forward(&input, &indices)?;
        let n_quantized = AtomicUsize::new(0);
        let converted = layer.apply_isq(
            Some(IsqType::Q4_0),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        let actual = converted.gather_forward(&input, &indices)?;

        assert_ne!(converted.name(), "mxfp4-layer");
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_close(&actual, &expected, 0.01)
    }

    #[test]
    fn stacked_mxfp4_preserves_packed_capture_and_exact_target() -> Result<()> {
        let source = stacked_test_layer()? as Arc<dyn QuantMethod>;
        let n_quantized = AtomicUsize::new(0);
        let captured = source.clone().apply_isq(
            None,
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        assert_eq!(captured.uqff_type(), Some(IsqType::MXFP4));
        assert!(captured.has_bias());

        let exact = source.apply_isq(
            Some(IsqType::MXFP4),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        assert_eq!(exact.uqff_type(), Some(IsqType::MXFP4));
        assert!(exact.has_bias());
        Ok(())
    }

    #[test]
    fn stacked_mxfp4_requantizes_to_afq_with_selected_bias() -> Result<()> {
        let layer = stacked_test_layer()?;
        let (input, indices) = stacked_test_inputs()?;
        let expected = layer.gather_forward(&input, &indices)?;
        let n_quantized = AtomicUsize::new(0);
        let converted = layer.apply_isq(
            Some(IsqType::AFQ4),
            Device::Cpu,
            &n_quantized,
            None,
            QuantizeOntoGuard::new(),
        )?;
        let actual = converted.gather_forward(&input, &indices)?;

        assert_ne!(converted.name(), "mxfp4-layer");
        assert_eq!(n_quantized.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_close(&actual, &expected, 0.01)
    }

    #[test]
    fn stacked_mxfp4_rejects_targets_without_expert_gather() -> Result<()> {
        let layer = stacked_test_layer()?;
        let request = crate::IsqRequest {
            ty: Some(IsqType::F8Q8),
            device: Device::Cpu,
            has_imatrix: false,
            capture: crate::IsqCaptureMode::Immediate,
            consumer: crate::IsqConsumer::UqffWrite,
            module_key: "experts".to_string(),
        };
        let plan_error = layer.plan_isq(&request).unwrap_err();
        assert!(plan_error
            .to_string()
            .contains("does not support stacked expert gather"));
        let error = layer
            .apply_isq(
                Some(IsqType::F8Q8),
                Device::Cpu,
                &AtomicUsize::new(0),
                None,
                QuantizeOntoGuard::new(),
            )
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("packed MXFP4 expert weights"));
        assert!(message.contains("does not support stacked expert gather"));
        Ok(())
    }
}
