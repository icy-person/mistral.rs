use std::{
    sync::{atomic::AtomicUsize, Arc, OnceLock},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _CMP_GT_OQ, __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256,
    _mm256_castps256_ps128, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps,
    _mm256_cmpgt_epi32, _mm256_cmpeq_epi32, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32,
    _mm256_cvttps_epi32, _mm256_div_ps, _mm256_extractf128_ps, _mm256_fmadd_ps,
    _mm256_hadd_ps, _mm256_rcp_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps,
    _mm256_or_si256, _mm256_permutevar8x32_ps, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setr_ps, _mm256_setzero_ps, _mm256_setzero_si256,
    _mm256_slli_epi32, _mm256_srli_epi32, _mm256_storeu_ps, _mm256_sub_epi32, _mm256_sub_ps, _mm256_xor_ps,
    _mm_add_ss, _mm_set1_epi16,
    _mm_and_si128, _mm_cvtss_f32, _mm_loadl_epi64, _mm_packus_epi16, _mm_setzero_si128,
    _mm_srli_epi16, _mm_unpacklo_epi8,
};

use candle_core::{DType, Device, Result, Storage, Tensor};
use rayon::prelude::*;
use safetensors::tensor::Dtype;

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::mxfp4_stream::{
    MxFp4StreamCache, MxFp4StreamData, MxFp4StreamKey, MxFp4StreamRange,
};
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
    cache_sources: Vec<Arc<str>>,
    num_experts: usize,
    component_out_dim: usize,
    in_dim: usize,
    out_dim: usize,
    bias: Option<Tensor>,
    bias_cpu: Option<Arc<Vec<f32>>>,
    expert_ranges: Vec<MxFp4StreamRange>,
    // Zero-copy entries are immutable mmap descriptors. Keep a tiny per-layer
    // OnceLock table so hot decode hits bypass the shared cache mutex/hash lookup.
    zero_copy_experts: Vec<Vec<OnceLock<Arc<MxFp4StreamData>>>,
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

        let cache_sources = raw_weights
            .iter()
            .map(|source| Arc::<str>::from(source.as_str()))
            .collect();

        let zero_copy_experts = (0..raw_weights.len())
            .map(|_| (0..num_experts).map(|_| OnceLock::new()).collect())
            .collect();

        Ok(Self {
            raw_weights,
            cache_sources,
            num_experts,
            component_out_dim,
            in_dim,
            out_dim,
            bias,
            bias_cpu,
            expert_ranges,
            zero_copy_experts,
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

    #[inline]
    fn remember_zero_copy_expert(
        &self,
        weight_idx: usize,
        expert_idx: usize,
        data: Arc<MxFp4StreamData>,
    ) -> Arc<MxFp4StreamData> {
        if !self.cache.zero_copy() {
            return data;
        }
        let slot = &self.zero_copy_experts[weight_idx][expert_idx];
        let _ = slot.set(data.clone());
        slot.get().cloned().unwrap_or(data)
    }

    #[inline]
    fn load_expert_cached(
        &self,
        weight_idx: usize,
        expert_idx: usize,
        key: &MxFp4StreamKey,
        range: MxFp4StreamRange,
    ) -> Result<Arc<MxFp4StreamData>> {
        if self.cache.zero_copy() {
            if let Some(data) = self.zero_copy_experts[weight_idx][expert_idx].get() {
                return Ok(data.clone());
            }
        }

        let data = self.cache.load(key, range)?;
        Ok(self.remember_zero_copy_expert(weight_idx, expert_idx, data))
    }

    #[cfg(target_os = "linux")]
    fn physical_core_count() -> Option<usize> {
        use std::{collections::HashSet, fs};

        let mut cores = HashSet::new();
        let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
            return None;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let path = entry.path().join("topology/core_id");
            if let Ok(core_id) = fs::read_to_string(path) {
                cores.insert(core_id.trim().to_string());
            }
        }
        (!cores.is_empty()).then_some(cores.len())
    }

    #[cfg(not(target_os = "linux"))]
    fn physical_core_count() -> Option<usize> {
        None
    }

    fn moe_thread_pool() -> &'static rayon::ThreadPool {
        static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let default_threads = candle_core::utils::get_num_threads().max(1);
            let threads = std::env::var("MISTRALRS_MOE_THREADS")
                .ok()
                .and_then(|value| {
                    if value.eq_ignore_ascii_case("physical") {
                        physical_core_count().or(Some(default_threads))
                    } else {
                        value.parse::<usize>().ok().filter(|&value| value > 0)
                    }
                })
                .unwrap_or(default_threads);
            let affinity = std::env::var("MISTRALRS_MOE_AFFINITY")
                .map(|value| !matches!(value.as_str(), "0" | "false" | "no"))
                .unwrap_or(true);

            let mut builder = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|idx| format!("mistralrs-moe-{idx}"));
            if affinity {
                builder = builder.start_handler(|_| candle_core::utils::set_thread_affinity());
            }
            builder
                .build()
                .expect("failed to build GPT-OSS MXFP4 CPU thread pool")
        })
    }

    #[inline(always)]
    fn adaptive_parallel_with_threads(
        threads: usize,
        route_count: usize,
        out_rows: usize,
        blocks_per_row: usize,
    ) -> bool {
        if threads <= 1 {
            return false;
        }
        let work = route_count
            .saturating_mul(out_rows)
            .saturating_mul(blocks_per_row);
        work >= 8192 && out_rows >= 32
    }

    #[inline(always)]
    fn adaptive_parallel(route_count: usize, out_rows: usize, blocks_per_row: usize) -> bool {
        MxFp4StreamingExpertLayer::adaptive_parallel_with_threads(
            rayon::current_num_threads(),
            route_count,
            out_rows,
            blocks_per_row,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_fused_weight_vectors_avx2(
        raw_expert: &[u8],
        block_start: usize,
    ) -> [__m256; 4] {
        // GPT-OSS MXFP4 uses E2M1 nibbles. Decode the 4-bit magnitudes with an
        // in-register AVX2 permutation instead of a random-access 4096-entry
        // gather table. The sign bit is carried by nibble bit 3.
        #[inline(always)]
        unsafe fn unpack8(ptr: *const u8) -> (__m256i, __m256i) {
            let bytes: __m128i = _mm_loadl_epi64(ptr.cast());
            let bytes16 = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
            let low16 = _mm_and_si128(bytes16, _mm_set1_epi16(0x000f));
            let high16 = _mm_and_si128(_mm_srli_epi16(bytes16, 4), _mm_set1_epi16(0x000f));
            let low8 = _mm_packus_epi16(low16, _mm_setzero_si128());
            let high8 = _mm_packus_epi16(high16, _mm_setzero_si128());
            (
                _mm256_cvtepu8_epi32(low8),
                _mm256_cvtepu8_epi32(high8),
            )
        }

        #[inline(always)]
        unsafe fn decode8_normal_e8m0(nibbles: __m256i, scale: __m256i) -> __m256 {
            // FP4 values are {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
            // Encode them directly as IEEE-754 bits:
            //   exponent adjustment: {0, -1, 0, 0, 1, 1, 2, 2}
            //   mantissa bit 22 set for {1.5, 3, 6}.
            let mag = _mm256_and_si256(nibbles, _mm256_set1_epi32(7));

            let hi = _mm256_srli_epi32(mag, 2);
            let hi2 = _mm256_srli_epi32(mag, 1);
            let high_adj = _mm256_add_epi32(
                hi,
                _mm256_and_si256(
                    hi,
                    _mm256_and_si256(hi2, _mm256_set1_epi32(1)),
                ),
            );
            let one = _mm256_cmpeq_epi32(mag, _mm256_set1_epi32(1));
            let exponent_adjust = _mm256_sub_epi32(
                high_adj,
                _mm256_and_si256(one, _mm256_set1_epi32(1)),
            );
            let exponent = _mm256_add_epi32(scale, exponent_adjust);
            let exponent_bits = _mm256_slli_epi32(exponent, 23);

            let gt_one = _mm256_cmpgt_epi32(mag, _mm256_set1_epi32(1));
            let odd = _mm256_and_si256(mag, _mm256_set1_epi32(1));
            let mantissa_bits = _mm256_slli_epi32(
                _mm256_and_si256(gt_one, odd),
                22,
            );

            let nonzero = _mm256_cmpgt_epi32(mag, _mm256_setzero_si256());
            let magnitude = _mm256_and_si256(
                _mm256_or_si256(exponent_bits, mantissa_bits),
                nonzero,
            );
            let sign = _mm256_slli_epi32(
                _mm256_and_si256(nibbles, _mm256_set1_epi32(8)),
                28,
            );
            _mm256_castsi256_ps(_mm256_or_si256(magnitude, sign))
        }

        #[inline(always)]
        unsafe fn decode8_fallback(nibbles: __m256i, scale: __m256) -> __m256 {
            // Preserve the exact reference behavior for subnormal and overflow edge
            // scales where direct exponent-field adjustment is not sufficient.
            let idx = _mm256_and_si256(nibbles, _mm256_set1_epi32(7));
            let magnitude = _mm256_permutevar8x32_ps(
                _mm256_setr_ps(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                idx,
            );
            let sign = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_and_si256(nibbles, _mm256_set1_epi32(8)),
                28,
            ));
            _mm256_xor_ps(_mm256_mul_ps(magnitude, scale), sign)
        }

        let packed = raw_expert.as_ptr().add(block_start + 1);
        let (lo0, hi0) = unpack8(packed);
        let (lo1, hi1) = unpack8(packed.add(8));

        let s = raw_expert[block_start] as u32;
        if (3..=253).contains(&s) {
            let scale_offset = _mm256_set1_epi32((s as i32) - 1);
            [
                decode8_normal_e8m0(lo0, scale_offset),
                decode8_normal_e8m0(lo1, scale_offset),
                decode8_normal_e8m0(hi0, scale_offset),
                decode8_normal_e8m0(hi1, scale_offset),
            ]
        } else {
            // E8M0 special values x=0,1 are the subnormal floor encodings.
            let scale_bits = if s < 2 {
                0x0020_0000u32 << s
            } else {
                (s - 1) << 23
            };
            let scale = _mm256_set1_ps(f32::from_bits(scale_bits));
            [
                decode8_fallback(lo0, scale),
                decode8_fallback(lo1, scale),
                decode8_fallback(hi0, scale),
                decode8_fallback(hi1, scale),
            ]
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot_streamed_row_fused_avx2(
        x: &[f32],
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
    ) -> f32 {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let mut weight_offset = row * row_bytes;
        let mut x_ptr = x.as_ptr();

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        for _ in 0..blocks_per_row {
            let w = Self::load_fused_weight_vectors_avx2(raw_expert, weight_offset);

            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(x_ptr), w[0]));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(8)), w[1]));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(16)), w[2]));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(24)), w[3]));

            weight_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            x_ptr = x_ptr.add(MXFP4_BLOCK_SIZE);
        }

        Self::hsum4(a0, a1, a2, a3)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot_streamed_row_fused_avx2_fma(
        x: &[f32],
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
    ) -> f32 {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let mut weight_offset = row * row_bytes;
        let mut x_ptr = x.as_ptr();

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        for _ in 0..blocks_per_row {
            let w = Self::load_fused_weight_vectors_avx2(raw_expert, weight_offset);

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), w[0], a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr.add(8)), w[1], a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr.add(16)), w[2], a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr.add(24)), w[3], a3);

            weight_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            x_ptr = x_ptr.add(MXFP4_BLOCK_SIZE);
        }

        Self::hsum4(a0, a1, a2, a3)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot_streamed_row_pair_fused_avx2(
        x: &[f32],
        raw0: &[u8],
        raw1: &[u8],
        row: usize,
        in_dim: usize,
    ) -> (f32, f32) {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let mut w0_offset = row * row_bytes;
        let mut w1_offset = row * row_bytes;
        let mut x_offset = 0usize;

        let mut g0 = _mm256_setzero_ps();
        let mut g1 = _mm256_setzero_ps();
        let mut g2 = _mm256_setzero_ps();
        let mut g3 = _mm256_setzero_ps();
        let mut u0 = _mm256_setzero_ps();
        let mut u1 = _mm256_setzero_ps();
        let mut u2 = _mm256_setzero_ps();
        let mut u3 = _mm256_setzero_ps();

        for _ in 0..blocks_per_row {
            let x0 = _mm256_loadu_ps(x.as_ptr().add(x_offset));
            let x1 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 8));
            let x2 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 16));
            let x3 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 24));

            let wg = Self::load_fused_weight_vectors_avx2(raw0, w0_offset);
            g0 = _mm256_add_ps(g0, _mm256_mul_ps(x0, wg[0]));
            g1 = _mm256_add_ps(g1, _mm256_mul_ps(x1, wg[1]));
            g2 = _mm256_add_ps(g2, _mm256_mul_ps(x2, wg[2]));
            g3 = _mm256_add_ps(g3, _mm256_mul_ps(x3, wg[3]));

            let wu = Self::load_fused_weight_vectors_avx2(raw1, w1_offset);
            u0 = _mm256_add_ps(u0, _mm256_mul_ps(x0, wu[0]));
            u1 = _mm256_add_ps(u1, _mm256_mul_ps(x1, wu[1]));
            u2 = _mm256_add_ps(u2, _mm256_mul_ps(x2, wu[2]));
            u3 = _mm256_add_ps(u3, _mm256_mul_ps(x3, wu[3]));

            w0_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            w1_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            x_offset += MXFP4_BLOCK_SIZE;
        }

        (Self::hsum4(g0, g1, g2, g3), Self::hsum4(u0, u1, u2, u3))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot_streamed_row_pair_fused_avx2_fma(
        x: &[f32],
        raw0: &[u8],
        raw1: &[u8],
        row: usize,
        in_dim: usize,
    ) -> (f32, f32) {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let mut w0_offset = row * row_bytes;
        let mut w1_offset = row * row_bytes;
        let mut x_offset = 0usize;

        let mut g0 = _mm256_setzero_ps();
        let mut g1 = _mm256_setzero_ps();
        let mut g2 = _mm256_setzero_ps();
        let mut g3 = _mm256_setzero_ps();
        let mut u0 = _mm256_setzero_ps();
        let mut u1 = _mm256_setzero_ps();
        let mut u2 = _mm256_setzero_ps();
        let mut u3 = _mm256_setzero_ps();

        for _ in 0..blocks_per_row {
            let x0 = _mm256_loadu_ps(x.as_ptr().add(x_offset));
            let x1 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 8));
            let x2 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 16));
            let x3 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 24));

            let wg = Self::load_fused_weight_vectors_avx2(raw0, w0_offset);
            g0 = _mm256_fmadd_ps(x0, wg[0], g0);
            g1 = _mm256_fmadd_ps(x1, wg[1], g1);
            g2 = _mm256_fmadd_ps(x2, wg[2], g2);
            g3 = _mm256_fmadd_ps(x3, wg[3], g3);

            let wu = Self::load_fused_weight_vectors_avx2(raw1, w1_offset);
            u0 = _mm256_fmadd_ps(x0, wu[0], u0);
            u1 = _mm256_fmadd_ps(x1, wu[1], u1);
            u2 = _mm256_fmadd_ps(x2, wu[2], u2);
            u3 = _mm256_fmadd_ps(x3, wu[3], u3);

            w0_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            w1_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            x_offset += MXFP4_BLOCK_SIZE;
        }

        (Self::hsum4(g0, g1, g2, g3), Self::hsum4(u0, u1, u2, u3))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot_streamed_row_interleaved_gate_up_fused_avx2(
        x: &[f32],
        raw_expert: &[u8],
        gate_row: usize,
        up_row: usize,
        in_dim: usize,
    ) -> (f32, f32) {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let mut gate_offset = gate_row * row_bytes;
        let mut up_offset = up_row * row_bytes;
        let mut x_offset = 0usize;

        let mut g0 = _mm256_setzero_ps();
        let mut g1 = _mm256_setzero_ps();
        let mut g2 = _mm256_setzero_ps();
        let mut g3 = _mm256_setzero_ps();
        let mut u0 = _mm256_setzero_ps();
        let mut u1 = _mm256_setzero_ps();
        let mut u2 = _mm256_setzero_ps();
        let mut u3 = _mm256_setzero_ps();

        for _ in 0..blocks_per_row {
            let x0 = _mm256_loadu_ps(x.as_ptr().add(x_offset));
            let x1 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 8));
            let x2 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 16));
            let x3 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 24));

            let wg = Self::load_fused_weight_vectors_avx2(raw_expert, gate_offset);
            g0 = _mm256_add_ps(g0, _mm256_mul_ps(x0, wg[0]));
            g1 = _mm256_add_ps(g1, _mm256_mul_ps(x1, wg[1]));
            g2 = _mm256_add_ps(g2, _mm256_mul_ps(x2, wg[2]));
            g3 = _mm256_add_ps(g3, _mm256_mul_ps(x3, wg[3]));

            let wu = Self::load_fused_weight_vectors_avx2(raw_expert, up_offset);
            u0 = _mm256_add_ps(u0, _mm256_mul_ps(x0, wu[0]));
            u1 = _mm256_add_ps(u1, _mm256_mul_ps(x1, wu[1]));
            u2 = _mm256_add_ps(u2, _mm256_mul_ps(x2, wu[2]));
            u3 = _mm256_add_ps(u3, _mm256_mul_ps(x3, wu[3]));

            gate_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            up_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            x_offset += MXFP4_BLOCK_SIZE;
        }

        (Self::hsum4(g0, g1, g2, g3), Self::hsum4(u0, u1, u2, u3))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot_streamed_row_interleaved_gate_up_fused_avx2_fma(
        x: &[f32],
        raw_expert: &[u8],
        gate_row: usize,
        up_row: usize,
        in_dim: usize,
    ) -> (f32, f32) {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let mut gate_offset = gate_row * row_bytes;
        let mut up_offset = up_row * row_bytes;
        let mut x_offset = 0usize;

        let mut g0 = _mm256_setzero_ps();
        let mut g1 = _mm256_setzero_ps();
        let mut g2 = _mm256_setzero_ps();
        let mut g3 = _mm256_setzero_ps();
        let mut u0 = _mm256_setzero_ps();
        let mut u1 = _mm256_setzero_ps();
        let mut u2 = _mm256_setzero_ps();
        let mut u3 = _mm256_setzero_ps();

        for _ in 0..blocks_per_row {
            let x0 = _mm256_loadu_ps(x.as_ptr().add(x_offset));
            let x1 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 8));
            let x2 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 16));
            let x3 = _mm256_loadu_ps(x.as_ptr().add(x_offset + 24));

            let wg = Self::load_fused_weight_vectors_avx2(raw_expert, gate_offset);
            g0 = _mm256_fmadd_ps(x0, wg[0], g0);
            g1 = _mm256_fmadd_ps(x1, wg[1], g1);
            g2 = _mm256_fmadd_ps(x2, wg[2], g2);
            g3 = _mm256_fmadd_ps(x3, wg[3], g3);

            let wu = Self::load_fused_weight_vectors_avx2(raw_expert, up_offset);
            u0 = _mm256_fmadd_ps(x0, wu[0], u0);
            u1 = _mm256_fmadd_ps(x1, wu[1], u1);
            u2 = _mm256_fmadd_ps(x2, wu[2], u2);
            u3 = _mm256_fmadd_ps(x3, wu[3], u3);

            gate_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            up_offset += MXFP4_BLOCK_SIZE / 2 + 1;
            x_offset += MXFP4_BLOCK_SIZE;
        }

        (Self::hsum4(g0, g1, g2, g3), Self::hsum4(u0, u1, u2, u3))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot_streamed_contiguous_routes_fused_avx2(
        x_data: &[f32],
        route_count: usize,
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
        accs: &mut [f32],
    ) {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let row_start = row * row_bytes;

        if route_count <= 4 {
            let mut lo = [_mm256_setzero_ps(); 4];
            let mut hi = [_mm256_setzero_ps(); 4];

            for block_idx in 0..blocks_per_row {
                let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                let w = Self::load_fused_weight_vectors_avx2(raw_expert, block_start);
                let x_start = block_idx * MXFP4_BLOCK_SIZE;

                for route_idx in 0..route_count {
                    let x = x_data.as_ptr().add(route_idx * in_dim + x_start);
                    lo[route_idx] = _mm256_add_ps(
                        lo[route_idx],
                        _mm256_mul_ps(_mm256_loadu_ps(x), w[0]),
                    );
                    lo[route_idx] = _mm256_add_ps(
                        lo[route_idx],
                        _mm256_mul_ps(_mm256_loadu_ps(x.add(8)), w[1]),
                    );
                    hi[route_idx] = _mm256_add_ps(
                        hi[route_idx],
                        _mm256_mul_ps(_mm256_loadu_ps(x.add(16)), w[2]),
                    );
                    hi[route_idx] = _mm256_add_ps(
                        hi[route_idx],
                        _mm256_mul_ps(_mm256_loadu_ps(x.add(24)), w[3]),
                    );
                }
            }

            for route_idx in 0..route_count {
                accs[route_idx] += Self::hsum2(lo[route_idx], hi[route_idx]);
            }
            return;
        }

        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let w = Self::load_fused_weight_vectors_avx2(raw_expert, block_start);
            let x_start = block_idx * MXFP4_BLOCK_SIZE;
            for route_idx in 0..route_count {
                let x = x_data.as_ptr().add(route_idx * in_dim + x_start);
                let a0 = _mm256_mul_ps(_mm256_loadu_ps(x), w[0]);
                let a1 = _mm256_mul_ps(_mm256_loadu_ps(x.add(8)), w[1]);
                let a2 = _mm256_mul_ps(_mm256_loadu_ps(x.add(16)), w[2]);
                let a3 = _mm256_mul_ps(_mm256_loadu_ps(x.add(24)), w[3]);
                accs[route_idx] += Self::hsum4(a0, a1, a2, a3);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot_streamed_contiguous_routes_fused_avx2_fma(
        x_data: &[f32],
        route_count: usize,
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
        accs: &mut [f32],
    ) {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let row_start = row * row_bytes;

        if route_count <= 4 {
            let mut lo = [_mm256_setzero_ps(); 4];
            let mut hi = [_mm256_setzero_ps(); 4];

            for block_idx in 0..blocks_per_row {
                let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                let w = Self::load_fused_weight_vectors_avx2(raw_expert, block_start);
                let x_start = block_idx * MXFP4_BLOCK_SIZE;

                for route_idx in 0..route_count {
                    let x = x_data.as_ptr().add(route_idx * in_dim + x_start);
                    lo[route_idx] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x), w[0], lo[route_idx]
                    );
                    lo[route_idx] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x.add(8)), w[1], lo[route_idx]
                    );
                    hi[route_idx] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x.add(16)), w[2], hi[route_idx]
                    );
                    hi[route_idx] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x.add(24)), w[3], hi[route_idx]
                    );
                }
            }

            for route_idx in 0..route_count {
                accs[route_idx] += Self::hsum2(lo[route_idx], hi[route_idx]);
            }
            return;
        }

        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let w = Self::load_fused_weight_vectors_avx2(raw_expert, block_start);
            let x_start = block_idx * MXFP4_BLOCK_SIZE;
            for route_idx in 0..route_count {
                let x = x_data.as_ptr().add(route_idx * in_dim + x_start);
                let a0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x), w[0], _mm256_setzero_ps()
                );
                let a1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.add(8)), w[1], _mm256_setzero_ps()
                );
                let a2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.add(16)), w[2], _mm256_setzero_ps()
                );
                let a3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.add(24)), w[3], _mm256_setzero_ps()
                );
                accs[route_idx] += Self::hsum4(a0, a1, a2, a3);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot_streamed_routes_fused_avx2(
        x_data: &[f32],
        x_offsets: &[usize],
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
        accs: &mut [f32],
    ) {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let row_start = row * row_bytes;

        if x_offsets.len() <= 4 {
            let mut v0 = [_mm256_setzero_ps(); 4];
            let mut v1 = [_mm256_setzero_ps(); 4];
            let n = x_offsets.len();

            for block_idx in 0..blocks_per_row {
                let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                let w = Self::load_fused_weight_vectors_avx2(raw_expert, block_start);
                let x_start = block_idx * MXFP4_BLOCK_SIZE;

                for (route_idx, &x_offset) in x_offsets.iter().enumerate() {
                    let x = x_data.as_ptr().add(x_offset + x_start);
                    v0[route_idx] = _mm256_add_ps(
                        v0[route_idx],
                        _mm256_mul_ps(_mm256_loadu_ps(x), w[0]),
                    );
                    v0[route_idx] = _mm256_add_ps(
                        v0[route_idx],
                        _mm256_mul_ps(_mm256_loadu_ps(x.add(8)), w[1]),
                    );
                    v1[route_idx] = _mm256_add_ps(
                        v1[route_idx],
                        _mm256_mul_ps(_mm256_loadu_ps(x.add(16)), w[2]),
                    );
                    v1[route_idx] = _mm256_add_ps(
                        v1[route_idx],
                        _mm256_mul_ps(_mm256_loadu_ps(x.add(24)), w[3]),
                    );
                }
            }

            for route_idx in 0..n {
                accs[route_idx] += Self::hsum2(v0[route_idx], v1[route_idx]);
            }
            return;
        }

        // For larger route groups, keep one weight decode shared across all routes.
        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let w = Self::load_fused_weight_vectors_avx2(raw_expert, block_start);
            let x_start = block_idx * MXFP4_BLOCK_SIZE;

            for (route_idx, &x_offset) in x_offsets.iter().enumerate() {
                let x = x_data.as_ptr().add(x_offset + x_start);
                let a0 = _mm256_mul_ps(_mm256_loadu_ps(x), w[0]);
                let a1 = _mm256_mul_ps(_mm256_loadu_ps(x.add(8)), w[1]);
                let a2 = _mm256_mul_ps(_mm256_loadu_ps(x.add(16)), w[2]);
                let a3 = _mm256_mul_ps(_mm256_loadu_ps(x.add(24)), w[3]);
                accs[route_idx] += Self::hsum4(a0, a1, a2, a3);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot_streamed_routes_fused_avx2_fma(
        x_data: &[f32],
        x_offsets: &[usize],
        raw_expert: &[u8],
        row: usize,
        in_dim: usize,
        accs: &mut [f32],
    ) {
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
        let row_start = row * row_bytes;

        if x_offsets.len() <= 4 {
            let mut lo = [_mm256_setzero_ps(); 4];
            let mut hi = [_mm256_setzero_ps(); 4];
            let n = x_offsets.len();

            for block_idx in 0..blocks_per_row {
                let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                let w = Self::load_fused_weight_vectors_avx2(raw_expert, block_start);
                let x_start = block_idx * MXFP4_BLOCK_SIZE;

                for (route_idx, &x_offset) in x_offsets.iter().enumerate() {
                    let x = x_data.as_ptr().add(x_offset + x_start);
                    lo[route_idx] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x), w[0], lo[route_idx]
                    );
                    lo[route_idx] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x.add(8)), w[1], lo[route_idx]
                    );
                    hi[route_idx] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x.add(16)), w[2], hi[route_idx]
                    );
                    hi[route_idx] = _mm256_fmadd_ps(
                        _mm256_loadu_ps(x.add(24)), w[3], hi[route_idx]
                    );
                }
            }

            for route_idx in 0..n {
                accs[route_idx] += Self::hsum2(lo[route_idx], hi[route_idx]);
            }
            return;
        }

        for block_idx in 0..blocks_per_row {
            let block_start = row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
            let w = Self::load_fused_weight_vectors_avx2(raw_expert, block_start);
            let x_start = block_idx * MXFP4_BLOCK_SIZE;

            for (route_idx, &x_offset) in x_offsets.iter().enumerate() {
                let x = x_data.as_ptr().add(x_offset + x_start);
                let a0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x), w[0], _mm256_setzero_ps()
                );
                let a1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.add(8)), w[1], _mm256_setzero_ps()
                );
                let a2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.add(16)), w[2], _mm256_setzero_ps()
                );
                let a3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.add(24)), w[3], _mm256_setzero_ps()
                );
                accs[route_idx] += Self::hsum4(a0, a1, a2, a3);
            }
        }
    }

    #[inline(always)]
    fn dot_block(x: &[f32], w: &[f32], _kernel: u8) -> f32 {
        debug_assert_eq!(x.len(), MXFP4_BLOCK_SIZE);
        debug_assert_eq!(w.len(), MXFP4_BLOCK_SIZE);

        let mut acc = 0f32;
        for i in 0..MXFP4_BLOCK_SIZE {
            acc += x[i] * w[i];
        }
        acc
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn hsum2(a0: __m256, a1: __m256) -> f32 {
        let sum = _mm256_add_ps(a0, a1);
        let h1 = _mm256_hadd_ps(sum, sum);
        let h2 = _mm256_hadd_ps(h1, h1);
        let lo = _mm256_castps256_ps128(h2);
        let hi = _mm256_extractf128_ps(h2, 1);
        _mm_cvtss_f32(_mm_add_ss(lo, hi))
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

    const GEMM_MIN_ROUTES: usize = 16;

    #[inline]
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
                    let dequant =
                        &MXFP4Layer::DEQUANT_LUT[raw_expert[block_start] as usize];
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

    #[inline]
    fn gemm_routes(
        x_data: &[f32],
        route_x_offsets: &[usize],
        raw_expert: &[u8],
        out_rows: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let route_count = route_x_offsets.len();
        if route_count < MxFp4StreamingExpertLayer::GEMM_MIN_ROUTES {
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

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn exp_ps_avx2(x: __m256) -> __m256 {
        // Cephes-style AVX2 exp approximation. Inputs are clamped to the range
        // where fp32 sigmoid is effectively saturated. This removes libm expf
        // calls from the per-element GPT-OSS SwiGLU hot path.
        let x = _mm256_max_ps(
            _mm256_min_ps(x, _mm256_set1_ps(88.376_262_664_794_9)),
            _mm256_set1_ps(-88.376_262_664_794_9),
        );

        const LOG2EF: f32 = 1.442_695_040_888_963_4;
        const C1: f32 = 0.693_359_375;
        const C2: f32 = -2.121_944_40e-4;

        let fx = _mm256_add_ps(
            _mm256_mul_ps(x, _mm256_set1_ps(LOG2EF)),
            _mm256_set1_ps(0.5),
        );
        let mut emm0 = _mm256_cvttps_epi32(fx);
        let tmp = _mm256_cvtepi32_ps(emm0);
        let mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OQ);
        emm0 = _mm256_sub_epi32(
            emm0,
            _mm256_and_si256(
                _mm256_castps_si256(mask),
                _mm256_set1_epi32(1),
            ),
        );

        let fx_i = _mm256_cvtepi32_ps(emm0);
        let mut r = _mm256_sub_ps(x, _mm256_mul_ps(fx_i, _mm256_set1_ps(C1)));
        r = _mm256_sub_ps(r, _mm256_mul_ps(fx_i, _mm256_set1_ps(C2)));

        let z = _mm256_mul_ps(r, r);
        let mut y = _mm256_set1_ps(1.987_569_15e-4);
        y = _mm256_add_ps(
            _mm256_mul_ps(y, r),
            _mm256_set1_ps(1.398_199_95e-3),
        );
        y = _mm256_add_ps(
            _mm256_mul_ps(y, r),
            _mm256_set1_ps(8.333_451_907_3e-3),
        );
        y = _mm256_add_ps(
            _mm256_mul_ps(y, r),
            _mm256_set1_ps(4.166_579_589_4e-2),
        );
        y = _mm256_add_ps(
            _mm256_mul_ps(y, r),
            _mm256_set1_ps(1.666_666_545_9e-1),
        );
        y = _mm256_add_ps(
            _mm256_mul_ps(y, r),
            _mm256_set1_ps(5.000_000_120_1e-1),
        );
        y = _mm256_add_ps(_mm256_mul_ps(y, z), _mm256_add_ps(r, _mm256_set1_ps(1.0)));

        emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
        emm0 = _mm256_slli_epi32(emm0, 23);
        y = _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
        y
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn swiglu8_avx2(gate: __m256, up: __m256, alpha: f32) -> __m256 {
        let scaled = _mm256_mul_ps(gate, _mm256_set1_ps(alpha));
        let neg = _mm256_sub_ps(_mm256_setzero_ps(), scaled);
        let exp_neg = exp_ps_avx2(neg);
        let denom = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg);
        // One Newton step after rcp_ps gives ~22 bits of reciprocal accuracy,
        // avoiding the high-latency scalar-precision vector divide in the
        // already-approximate SwiGLU path.
        let mut sigmoid = _mm256_rcp_ps(denom);
        sigmoid = _mm256_mul_ps(
            sigmoid,
            _mm256_sub_ps(_mm256_set1_ps(2.0), _mm256_mul_ps(denom, sigmoid)),
        );
        _mm256_mul_ps(
            _mm256_mul_ps(_mm256_add_ps(up, _mm256_set1_ps(1.0)), gate),
            sigmoid,
        )
    }

    #[inline]
    fn approx_swiglu_enabled() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            static ENABLED: OnceLock<bool> = OnceLock::new();
            *ENABLED.get_or_init(|| {
                std::env::var("MISTRALRS_MOE_APPROX_SWIGLU")
                    .map(|value| !matches!(value.as_str(), "0" | "false" | "no"))
                    .unwrap_or(true)
            })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Fused CPU GPT-OSS path for native streamed MXFP4 experts.
    ///
    /// Computes gate/up, GPT-OSS SwiGLU, down projection, and top-k weighted
    /// reduction without materializing Tensor intermediates.
    pub(crate) fn fused_gptoss_mlp(
        gate_up: &MxFp4StreamingExpertLayer,
        down: &MxFp4StreamingExpertLayer,
        x: &Tensor,
        indices: &Tensor,
        weights: &Tensor,
        alpha: f32,
        limit: f32,
    ) -> Result<Option<Tensor>> {
        let (num_tokens, hidden_dim) = x.dims2()?;
        let (idx_tokens, topk) = indices.dims2()?;
        if idx_tokens != num_tokens || topk == 0 || weights.dims2()? != (num_tokens, topk) {
            return Ok(None);
        }
        // Whole-MLP fusion is tuned for autoregressive decode. Larger token
        // batches stay on the existing GEMM/routed path, which is usually better
        // for prompt processing on CPU.
        let fused_route_limit = std::env::var("MISTRALRS_MOE_FUSED_ROUTE_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(16);
        if num_tokens.saturating_mul(topk) > fused_route_limit {
            return Ok(None);
        }

        if gate_up.num_experts != down.num_experts
            || gate_up.raw_weights.len() != 1
            || down.raw_weights.len() != 1
            || gate_up.out_dim != gate_up.component_out_dim
            || gate_up.component_out_dim != down.in_dim * 2
            || gate_up.in_dim != hidden_dim
            || down.component_out_dim != down.out_dim
            || down.out_dim != hidden_dim
            || !gate_up.in_dim.is_multiple_of(MXFP4_BLOCK_SIZE)
            || !down.in_dim.is_multiple_of(MXFP4_BLOCK_SIZE)
            || MxFp4StreamingExpertLayer::dot_kernel() == 0
        {
            return Ok(None);
        }

        let x_cpu = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let x_data = x_cpu.flatten_all()?.to_vec1::<f32>()?;
        let ids_cpu = indices.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;
        let ids = ids_cpu.flatten_all()?.to_vec1::<u32>()?;
        let w_cpu = weights.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let route_weights = w_cpu.flatten_all()?.to_vec1::<f32>()?;

        let mut expert_counts = vec![0usize; gate_up.num_experts];
        for &id in &ids {
            let expert = id as usize;
            if expert >= gate_up.num_experts {
                candle_core::bail!("fused GPT-OSS expert index {expert} out of range");
            }
            expert_counts[expert] += 1;
        }

        let mut expert_offsets = vec![0usize; gate_up.num_experts + 1];
        for expert in 0..gate_up.num_experts {
            expert_offsets[expert + 1] = expert_offsets[expert] + expert_counts[expert];
        }

        let mut routes_flat = vec![0usize; ids.len()];
        expert_counts[..gate_up.num_experts].copy_from_slice(&expert_offsets[..gate_up.num_experts]);
        for (route, &id) in ids.iter().enumerate() {
            let expert = id as usize;
            let dst = expert_counts[expert];
            routes_flat[dst] = route;
            expert_counts[expert] += 1;
        }

        let mut experts = (0..gate_up.num_experts)
            .filter(|&expert| expert_offsets[expert] != expert_offsets[expert + 1])
            .collect::<Vec<_>>();

        if gate_up.cache.zero_copy() {
            experts.sort_unstable();
        } else {
            experts.sort_unstable_by_key(|&expert| {
                expert_offsets[expert + 1] - expert_offsets[expert]
            });
        }

        // Preserve the existing streamed GEMM fast path for large expert groups.
        if experts.iter().any(|&expert| {
            expert_offsets[expert + 1] - expert_offsets[expert] >= MxFp4StreamingExpertLayer::GEMM_MIN_ROUTES
        }) {
            return Ok(None);
        }

        let shared_cache = Arc::ptr_eq(&gate_up.cache, &down.cache);
        let mut requests = Vec::with_capacity(experts.len() * 2);
        if shared_cache && gate_up.cache.overlap() {
            for &expert in &experts {
                requests.push((
                    MxFp4StreamKey {
                        source: gate_up.cache_sources[0].clone(),
                        expert_index: expert,
                    },
                    gate_up.raw_expert_range(0, expert)?,
                ));
                requests.push((
                    MxFp4StreamKey {
                        source: down.cache_sources[0].clone(),
                        expert_index: expert,
                    },
                    down.raw_expert_range(0, expert)?,
                ));
            }
        }

        let mut pending = if shared_cache && gate_up.cache.overlap() {
            Some(gate_up.cache.prefetch(&requests)?)
        } else {
            None
        };

        let gate_bias = gate_up.bias_cpu.as_deref();
        let down_bias = down.bias_cpu.as_deref();
        let mut output = vec![0.0f32; num_tokens * hidden_dim];
        let mut activations = Vec::new();
        let kernel = MxFp4StreamingExpertLayer::dot_kernel();
        let approx_swiglu = approx_swiglu_enabled();
        let moe_threads = MxFp4StreamingExpertLayer::moe_thread_pool().current_num_threads();
        for &expert_idx in &experts {
            let (gate_data, down_data) = if let Some(queue) = pending.as_mut() {
                let gate_key = MxFp4StreamKey {
                    source: gate_up.raw_weights[0].clone(),
                    expert_index: expert_idx,
                };
                let gate_handle = queue.pop_front().ok_or_else(|| {
                    candle_core::Error::Msg("fused GPT-OSS gate request queue underflow".into())
                })?;
                let gate_data = gate_up
                    .remember_zero_copy_expert(
                        0,
                        expert_idx,
                        gate_up.cache.resolve(&gate_key, gate_handle)?,
                    );

                let down_key = MxFp4StreamKey {
                    source: down.raw_weights[0].clone(),
                    expert_index: expert_idx,
                };
                let down_handle = queue.pop_front().ok_or_else(|| {
                    candle_core::Error::Msg("fused GPT-OSS down request queue underflow".into())
                })?;
                let down_data = down.remember_zero_copy_expert(
                    0,
                    expert_idx,
                    down.cache.resolve(&down_key, down_handle)?,
                );
                (gate_data, down_data)
            } else {
                let gate_key = MxFp4StreamKey {
                    source: gate_up.raw_weights[0].clone(),
                    expert_index: expert_idx,
                };
                let down_key = MxFp4StreamKey {
                    source: down.raw_weights[0].clone(),
                    expert_index: expert_idx,
                };
                (
                    gate_up.load_expert_cached(
                        0,
                        expert_idx,
                        &gate_key,
                        gate_up.raw_expert_range(0, expert_idx)?,
                    )?,
                    down.load_expert_cached(
                        0,
                        expert_idx,
                        &down_key,
                        down.raw_expert_range(0, expert_idx)?,
                    )?,
                )
            };

            let start = expert_offsets[expert_idx];
            let end = expert_offsets[expert_idx + 1];
            let routes = &routes_flat[start..end];
            let route_count = routes.len();

            let activation_len = route_count * down.in_dim;
            activations.resize(activation_len, 0.0);

            let gate_raw = gate_data.as_ref();
            let down_raw = down_data.as_ref();
            let gate_bias_expert = gate_bias.map(|bias| &bias[expert_idx * gate_up.out_dim..]);

            // Full graph fusion: gate/up dot -> clamp -> SwiGLU directly into
            // activation scratch. No persistent gate_values/up_values tensors.
            MxFp4StreamingExpertLayer::moe_thread_pool().install(|| {
                activations[..activation_len]
                    .par_chunks_mut(8)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let base_flat = chunk_idx * 8;
                        let mut gates = [0.0f32; 8];
                        let mut ups = [0.0f32; 8];

                        for lane in 0..out_chunk.len() {
                            let flat_idx = base_flat + lane;
                            let route_idx = flat_idx / down.in_dim;
                            let row = flat_idx % down.in_dim;
                            let token = routes[route_idx] / topk;
                            let x_offset = token * hidden_dim;
                            let x_row = &x_data[x_offset..x_offset + gate_up.in_dim];

                            let (gate_value, up_value) = if kernel >= 2 {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    MxFp4StreamingExpertLayer::dot_streamed_row_interleaved_gate_up_fused_avx2_fma(
                                        x_row,
                                        gate_raw,
                                        row * 2,
                                        row * 2 + 1,
                                        gate_up.in_dim,
                                    )
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    unreachable!()
                                }
                            } else {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    MxFp4StreamingExpertLayer::dot_streamed_row_interleaved_gate_up_fused_avx2(
                                        x_row,
                                        gate_raw,
                                        row * 2,
                                        row * 2 + 1,
                                        gate_up.in_dim,
                                    )
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    unreachable!()
                                }
                            };

                            gates[lane] = gate_bias_ref
                                .map(|bias| {
                                    gate_value
                                        + bias[expert_idx * gate_up.out_dim + row * 2]
                                })
                                .unwrap_or(gate_value)
                                .min(limit);
                            ups[lane] = gate_bias_expert
                                .map(|bias| up_value + bias[row * 2 + 1])
                                .unwrap_or(up_value)
                                .clamp(-limit, limit);
                        }

                        if approx_swiglu && out_chunk.len() == 8 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                let g = _mm256_loadu_ps(gates.as_ptr());
                                let u = _mm256_loadu_ps(ups.as_ptr());
                                let y = swiglu8_avx2(g, u, alpha);
                                _mm256_storeu_ps(out_chunk.as_mut_ptr(), y);
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            unreachable!();
                        } else {
                            for lane in 0..out_chunk.len() {
                                out_chunk[lane] = (ups[lane] + 1.0)
                                    * gates[lane]
                                    / (1.0 + (-gates[lane] * alpha).exp());
                            }
                        }
                    });
            });

            // Top-k weighting is folded into the final down-projection store.
            // Do not traverse the full activation buffer just to scale it.
            let blocks_per_row = down.in_dim / MXFP4_BLOCK_SIZE;
            let down_kernel = kernel;

            const STACK_ROUTES: usize = 8;

            // Decode normally has one route per active expert. Avoid the generic
            // route-array kernel in that case: one direct row dot is cheaper and
            // needs no per-row accumulator array.
            if route_count == 1 {
                let route_row = routes[0];
                let token = route_row / topk;
                let weight = route_weights[route_row];
                let activation = &activations[..down.in_dim];
                let output_ptr = output.as_mut_ptr();
                let down_bias_ref = down_bias;

                let compute_down_row = |row: usize| {
                    let value = if down_kernel >= 2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            MxFp4StreamingExpertLayer::dot_streamed_row_fused_avx2_fma(
                                activation,
                                down_raw,
                                row,
                                down.in_dim,
                            )
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            unreachable!()
                        }
                    } else {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            MxFp4StreamingExpertLayer::dot_streamed_row_fused_avx2(
                                activation,
                                down_raw,
                                row,
                                down.in_dim,
                            )
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            unreachable!()
                        }
                    };
                    let bias = down_bias_ref
                        .map(|b| b[expert_idx * down.out_dim + row])
                        .unwrap_or(0.0);
                    unsafe {
                        *output_ptr.add(token * hidden_dim + row) +=
                            (value + bias) * weight;
                    }
                };

                if MxFp4StreamingExpertLayer::adaptive_parallel_with_threads(
                    moe_threads,
                    1,
                    down.out_dim,
                    blocks_per_row,
                ) {
                    MxFp4StreamingExpertLayer::moe_thread_pool().install(|| {
                        (0..down.out_dim)
                            .into_par_iter()
                            .for_each(compute_down_row);
                    });
                } else {
                    for row in 0..down.out_dim {
                        compute_down_row(row);
                    }
                }
                continue;
            }

            let parallel_down = route_count <= STACK_ROUTES
                && MxFp4StreamingExpertLayer::adaptive_parallel_with_threads(
                    moe_threads,
                    route_count,
                    down.out_dim,
                    blocks_per_row,
                );

            if parallel_down {
                let output_ptr = output.as_mut_ptr();
                let down_bias_ref = down_bias;
                MxFp4StreamingExpertLayer::moe_thread_pool().install(|| {
                    (0..down.out_dim).into_par_iter().for_each(|row| {
                    let mut accs = [0.0f32; STACK_ROUTES];
                    let accs = &mut accs[..route_count];
                    if down_kernel >= 2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            MxFp4StreamingExpertLayer::dot_streamed_contiguous_routes_fused_avx2_fma(
                                &activations,
                                route_count,
                                down_raw,
                                row,
                                down.in_dim,
                                accs,
                            );
                        }
                    } else {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            MxFp4StreamingExpertLayer::dot_streamed_contiguous_routes_fused_avx2(
                                &activations,
                                route_count,
                                down_raw,
                                row,
                                down.in_dim,
                                accs,
                            );
                        }
                    }

                    for route_idx in 0..route_count {
                        let route_row = routes[route_idx];
                        let token = route_row / topk;
                        let weight = route_weights[route_row];
                        let bias = down_bias_ref
                            .map(|b| b[expert_idx * down.out_dim + row])
                            .unwrap_or(0.0);
                        unsafe {
                            *output_ptr.add(token * hidden_dim + row) +=
                                (accs[route_idx] + bias) * weight;
                        }
                    }
                    });
                });
            } else {
                let mut accs = vec![0.0f32; route_count];
                for row in 0..down.out_dim {
                    accs.fill(0.0);
                    if down_kernel >= 2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            MxFp4StreamingExpertLayer::dot_streamed_contiguous_routes_fused_avx2_fma(
                                &activations,
                                route_count,
                                down_raw,
                                row,
                                down.in_dim,
                                &mut accs,
                            );
                        }
                    } else {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            MxFp4StreamingExpertLayer::dot_streamed_contiguous_routes_fused_avx2(
                                &activations,
                                route_count,
                                down_raw,
                                row,
                                down.in_dim,
                                &mut accs,
                            );
                        }
                    }
                    for (route_idx, &route_row) in routes.iter().enumerate() {
                        let token = route_row / topk;
                        let weight = route_weights[route_row];
                        let bias = down_bias
                            .map(|b| b[expert_idx * down.out_dim + row])
                            .unwrap_or(0.0);
                        output[token * hidden_dim + row] +=
                            weight * (accs[route_idx] + bias);
                    }
                }
            }
        }

        gate_up.cache.log_stats();
        if !shared_cache {
            down.cache.log_stats();
        }

        Ok(Some(
            Tensor::from_vec(output, (num_tokens, hidden_dim), &Device::Cpu)?
                .to_device(x.device())?
                .to_dtype(x.dtype())?,
        ))
    }

impl QuantMethod for MxFp4StreamingExpertLayer {
    fn as_mxfp4_streaming(&self) -> Option<&crate::MxFp4StreamingExpertLayer> {
        Some(self)
    }

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

        if indices_data.len() != num_tokens * topk {
            candle_core::bail!(
                "GPT-OSS MXFP4 streaming indices have {} routes, expected {}",
                indices_data.len(),
                num_tokens * topk
            );
        }

        // Flat routing representation: one contiguous route buffer plus expert
        // prefix offsets. This removes one heap allocation per expert from the old
        // Vec<Vec<usize>> representation.
        let mut expert_counts = vec![0usize; self.num_experts];
        for &expert in &indices_data {
            let expert = expert as usize;
            if expert >= self.num_experts {
                candle_core::bail!(
                    "GPT-OSS MXFP4 expert index {expert} out of range for {} experts",
                    self.num_experts
                );
            }
            expert_counts[expert] += 1;
        }

        let mut expert_offsets = vec![0usize; self.num_experts + 1];
        for expert in 0..self.num_experts {
            expert_offsets[expert + 1] =
                expert_offsets[expert].saturating_add(expert_counts[expert]);
        }

        let mut routes_flat = vec![0usize; indices_data.len()];
        // Reuse expert_counts as the write cursor after the prefix offsets are built;
        // this removes one temporary Vec allocation from every routed forward.
        expert_counts[..self.num_experts].copy_from_slice(&expert_offsets[..self.num_experts]);
        for (route, &expert) in indices_data.iter().enumerate() {
            let expert = expert as usize;
            let dst = expert_counts[expert];
            routes_flat[dst] = route;
            expert_counts[expert] += 1;
        }

        let mut experts = (0..self.num_experts)
            .filter(|&expert| expert_offsets[expert] != expert_offsets[expert + 1])
            .collect::<Vec<_>>();

        if self.cache.zero_copy() {
            // Preserve GGUF file locality in zero-copy mode.
            experts.sort_unstable();
        } else {
            // In heap-backed fallback mode, load cold experts first so the
            // per-source quota leaves hot experts resident.
            experts.sort_unstable_by_key(|&expert| {
                (
                    expert_offsets[expert + 1].saturating_sub(expert_offsets[expert]),
                    expert,
                )
            });
        }

        let mut requests = Vec::with_capacity(experts.len() * self.raw_weights.len());
        // Expert-major ordering keeps gate/up requests adjacent, allowing the
        // single-route GPT-OSS fast path to consume both handles in one pass.
        for &expert_idx in &experts {
            for weight_idx in 0..self.raw_weights.len() {
                requests.push((
                    MxFp4StreamKey {
                        source: self.cache_sources[weight_idx].clone(),
                        expert_index: expert_idx,
                    },
                    self.raw_expert_range(weight_idx, expert_idx)?,
                ));
            }
        }

        let mut pending = if self.cache.overlap() {
            Some(self.cache.prefetch(&requests)?)
        } else {
            None
        };

        let mut output = vec![0f32; num_tokens * topk * self.out_dim];
        if let Some(bias_data) = &self.bias_cpu {
            output
                .par_chunks_mut(self.out_dim)
                .enumerate()
                .for_each(|(route_row, out_row)| {
                    let expert_idx = indices_data[route_row] as usize;
                    let bias_offset = expert_idx * self.out_dim;
                    out_row.copy_from_slice(&bias_data[bias_offset..bias_offset + self.out_dim]);
                });
        }
        let kernel = Self::dot_kernel();

        // Reused across all experts/components in this forward.
        let mut route_x_offsets = Vec::<usize>::new();
        let mut partial = Vec::<f32>::new();
        let interleaved_gate_up = self.raw_weights.len() == 2;

        for component in 0..self.raw_weights.len() {
            for &expert_idx in &experts {
                let route_start = expert_offsets[expert_idx];
                let route_end = expert_offsets[expert_idx + 1];
                let routes = &routes_flat[route_start..route_end];
                let route_count = routes.len();
                let pair_single_route = interleaved_gate_up && route_count == 1;

                // Component 0 owns the paired gate/up fast path. Component 1 is
                // skipped only when component 0 already consumed both experts.
                if component == 1 && pair_single_route {
                    continue;
                }

                route_x_offsets.clear();
                if route_x_offsets.capacity() < route_count {
                    route_x_offsets.reserve(route_count - route_x_offsets.capacity());
                }
                for &route_row in routes {
                    route_x_offsets.push(if x_has_topk {
                        route_row * self.in_dim
                    } else {
                        (route_row / topk) * self.in_dim
                    });
                }

                let key = MxFp4StreamKey {
                    source: self.cache_sources[component].clone(),
                    expert_index: expert_idx,
                };
                let expert_data = if let Some(pending_queue) = pending.as_mut() {
                    let handle = pending_queue.pop_front().ok_or_else(|| {
                        candle_core::Error::Msg(
                            "GPT-OSS MXFP4 streamed expert request was not scheduled"
                                .to_string(),
                        )
                    })?;
                    self.remember_zero_copy_expert(
                        component,
                        expert_idx,
                        self.cache.resolve(&key, handle)?,
                    )
                } else {
                    let range = self.raw_expert_range(component, expert_idx)?;
                    self.load_expert_cached(component, expert_idx, &key, range)?
                };
                let expert = expert_data.as_ref();

                let blocks_per_row = self.in_dim / MXFP4_BLOCK_SIZE;

                if pair_single_route && component == 0 {
                    let key_up = MxFp4StreamKey {
                        source: self.cache_sources[1].clone(),
                        expert_index: expert_idx,
                    };
                    let expert_up_data = if let Some(pending_queue) = pending.as_mut() {
                        let handle = pending_queue.pop_front().ok_or_else(|| {
                            candle_core::Error::Msg(
                                "GPT-OSS MXFP4 paired gate/up request was not scheduled"
                                    .to_string(),
                            )
                        })?;
                        self.cache.resolve(&key_up, handle)?
                    } else {
                        let range = self.raw_expert_range(1, expert_idx)?;
                        self.cache.load(&key_up, range)?
                    };
                    let expert_up = expert_up_data.as_ref();
                    let x_row = &x_data[route_x_offsets[0]..route_x_offsets[0] + self.in_dim];
                    let out_row = &mut output[routes[0] * self.out_dim..(routes[0] + 1) * self.out_dim];

                    let compute_pair = |row: usize, pair: &mut [f32]| {
                        let (gate, up) = if kernel >= 2 {
                            #[cfg(target_arch = "x86_64")]
                            {
                                unsafe {
                                    Self::dot_streamed_row_pair_fused_avx2_fma(
                                        x_row,
                                        expert,
                                        expert_up,
                                        row,
                                        self.in_dim,
                                    )
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                (0.0, 0.0)
                            }
                        } else if kernel == 1 {
                            #[cfg(target_arch = "x86_64")]
                            {
                                unsafe {
                                    Self::dot_streamed_row_pair_fused_avx2(
                                        x_row,
                                        expert,
                                        expert_up,
                                        row,
                                        self.in_dim,
                                    )
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                (0.0, 0.0)
                            }
                        } else {
                            let mut gate = 0.0f32;
                            let mut up = 0.0f32;
                            let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                            for block_idx in 0..blocks_per_row {
                                let block_start = row * row_bytes
                                    + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                                let dequant_gate = &MXFP4Layer::DEQUANT_LUT[expert[block_start] as usize];
                                let dequant_up = &MXFP4Layer::DEQUANT_LUT[expert_up[block_start] as usize];
                                let packed_gate = &expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                let packed_up = &expert_up[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                let col_start = block_idx * MXFP4_BLOCK_SIZE;
                                for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                    let pg = packed_gate[byte_idx];
                                    let pu = packed_up[byte_idx];
                                    gate += x_row[col_start + byte_idx] * dequant_gate[(pg & 0x0f) as usize];
                                    gate += x_row[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx] * dequant_gate[(pg >> 4) as usize];
                                    up += x_row[col_start + byte_idx] * dequant_up[(pu & 0x0f) as usize];
                                    up += x_row[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx] * dequant_up[(pu >> 4) as usize];
                                }
                            }
                            (gate, up)
                        };
                        pair[0] += gate;
                        pair[1] += up;
                    };

                    let parallel =
                        Self::adaptive_parallel(1, self.component_out_dim, blocks_per_row);
                    if parallel {
                        out_row
                            .par_chunks_mut(2)
                            .enumerate()
                            .for_each(|(row, pair)| compute_pair(row, pair));
                    } else {
                        for (row, pair) in out_row.chunks_mut(2).enumerate() {
                            compute_pair(row, pair);
                        }
                    }
                    continue;
                }

                if route_count >= Self::GEMM_MIN_ROUTES {
                    let result = Self::gemm_routes(
                        &x_data,
                        &route_x_offsets,
                        expert,
                        self.component_out_dim,
                        self.in_dim,
                    )?;

                    for (route_idx, &route_row) in routes.iter().enumerate() {
                        let out_row =
                            &mut output[route_row * self.out_dim..(route_row + 1) * self.out_dim];
                        let base = route_idx * self.component_out_dim;
                        for row in 0..self.component_out_dim {
                            let col = if interleaved_gate_up {
                                row * 2 + component
                            } else {
                                row
                            };
                            out_row[col] += result[base + row];
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

                    let parallel = Self::adaptive_parallel(1, self.component_out_dim, blocks_per_row);
                    if parallel {
                        if interleaved_gate_up {
                            out_row
                                .par_chunks_mut(2)
                                .enumerate()
                                .for_each(|(row, pair)| {
                                    let value = if kernel >= 2 {
                                        #[cfg(target_arch = "x86_64")]
                                        {
                                            unsafe {
                                                Self::dot_streamed_row_fused_avx2_fma(
                                                    x_row,
                                                    expert,
                                                    row,
                                                    self.in_dim,
                                                )
                                            }
                                        }
                                        #[cfg(not(target_arch = "x86_64"))]
                                        {
                                            0.0
                                        }
                                    } else if kernel == 1 {
                                        #[cfg(target_arch = "x86_64")]
                                        {
                                            unsafe {
                                                Self::dot_streamed_row_fused_avx2(
                                                    x_row,
                                                    expert,
                                                    row,
                                                    self.in_dim,
                                                )
                                            }
                                        }
                                        #[cfg(not(target_arch = "x86_64"))]
                                        {
                                            0.0
                                        }
                                    } else {
                                        let row_bytes =
                                            blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                                        let mut acc = 0.0f32;
                                        for block_idx in 0..blocks_per_row {
                                            let block_start =
                                                row * row_bytes
                                                    + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                                            let scale =
                                                &MXFP4Layer::DEQUANT_LUT[expert[block_start] as usize];
                                            let packed = &expert[block_start + 1
                                                ..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                            let col_start = block_idx * MXFP4_BLOCK_SIZE;
                                            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                                let packed_byte = packed[byte_idx];
                                                acc += x_row[col_start + byte_idx]
                                                    * scale[(packed_byte & 0x0f) as usize];
                                                acc += x_row[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx]
                                                    * scale[(packed_byte >> 4) as usize];
                                            }
                                        }
                                        acc
                                    };
                                    pair[component] += value;
                                });
                        } else {
                            out_row
                                .par_iter_mut()
                                .enumerate()
                                .for_each(|(row, value)| {
                                    let dot = if kernel >= 2 {
                                        #[cfg(target_arch = "x86_64")]
                                        {
                                            unsafe {
                                                Self::dot_streamed_row_fused_avx2_fma(
                                                    x_row,
                                                    expert,
                                                    row,
                                                    self.in_dim,
                                                )
                                            }
                                        }
                                        #[cfg(not(target_arch = "x86_64"))]
                                        {
                                            0.0
                                        }
                                    } else if kernel == 1 {
                                        #[cfg(target_arch = "x86_64")]
                                        {
                                            unsafe {
                                                Self::dot_streamed_row_fused_avx2(
                                                    x_row,
                                                    expert,
                                                    row,
                                                    self.in_dim,
                                                )
                                            }
                                        }
                                        #[cfg(not(target_arch = "x86_64"))]
                                        {
                                            0.0
                                        }
                                    } else {
                                        let row_bytes =
                                            blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                                        let mut acc = 0.0f32;
                                        for block_idx in 0..blocks_per_row {
                                            let block_start =
                                                row * row_bytes
                                                    + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                                            let scale =
                                                &MXFP4Layer::DEQUANT_LUT[expert[block_start] as usize];
                                            let packed = &expert[block_start + 1
                                                ..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                            let col_start = block_idx * MXFP4_BLOCK_SIZE;
                                            for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                                let packed_byte = packed[byte_idx];
                                                acc += x_row[col_start + byte_idx]
                                                    * scale[(packed_byte & 0x0f) as usize];
                                                acc += x_row[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx]
                                                    * scale[(packed_byte >> 4) as usize];
                                            }
                                        }
                                        acc
                                    };
                                    *value += dot;
                                });
                        }
                    } else {
                        for row in 0..self.component_out_dim {
                            let value = if kernel >= 2 {
                                #[cfg(target_arch = "x86_64")]
                                {
                                    unsafe {
                                        Self::dot_streamed_row_fused_avx2_fma(
                                            x_row,
                                            expert,
                                            row,
                                            self.in_dim,
                                        )
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    let row_bytes =
                                        blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                                    let mut acc = 0.0f32;
                                    for block_idx in 0..blocks_per_row {
                                        let block_start =
                                            row * row_bytes
                                                + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                                        let scale = &MXFP4Layer::DEQUANT_LUT
                                            [expert[block_start] as usize];
                                        let packed = &expert[block_start + 1
                                            ..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                        let col_start = block_idx * MXFP4_BLOCK_SIZE;
                                        for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                            let packed_byte = packed[byte_idx];
                                            acc += x_row[col_start + byte_idx]
                                                * scale[(packed_byte & 0x0f) as usize];
                                            acc += x_row[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx]
                                                * scale[(packed_byte >> 4) as usize];
                                        }
                                    }
                                    acc
                                }
                            } else if kernel == 1 {
                                #[cfg(target_arch = "x86_64")]
                                {
                                    unsafe {
                                        Self::dot_streamed_row_fused_avx2(
                                            x_row,
                                            expert,
                                            row,
                                            self.in_dim,
                                        )
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    let row_bytes =
                                        blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                                    let mut acc = 0.0f32;
                                    for block_idx in 0..blocks_per_row {
                                        let block_start =
                                            row * row_bytes
                                                + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                                        let scale = &MXFP4Layer::DEQUANT_LUT
                                            [expert[block_start] as usize];
                                        let packed = &expert[block_start + 1
                                            ..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                        let col_start = block_idx * MXFP4_BLOCK_SIZE;
                                        for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                            let packed_byte = packed[byte_idx];
                                            acc += x_row[col_start + byte_idx]
                                                * scale[(packed_byte & 0x0f) as usize];
                                            acc += x_row[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx]
                                                * scale[(packed_byte >> 4) as usize];
                                        }
                                    }
                                    acc
                                }
                            } else {
                                let mut acc = 0.0f32;
                                let row_bytes =
                                    blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                                for block_idx in 0..blocks_per_row {
                                    let block_start =
                                        row * row_bytes
                                            + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                                    let scale =
                                        &MXFP4Layer::DEQUANT_LUT[expert[block_start] as usize];
                                    let packed = &expert[block_start + 1
                                        ..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                    let col_start = block_idx * MXFP4_BLOCK_SIZE;
                                    for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                        let packed_byte = packed[byte_idx];
                                        acc += x_row[col_start + byte_idx]
                                            * scale[(packed_byte & 0x0f) as usize];
                                        acc += x_row[col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx]
                                            * scale[(packed_byte >> 4) as usize];
                                    }
                                }
                                acc
                            };

                            let col = if interleaved_gate_up {
                                row * 2 + component
                            } else {
                                row
                            };
                            out_row[col] += value;
                        }
                    }
                    continue;
                }

                let scratch_len = self.component_out_dim * route_count;
                if partial.len() < scratch_len {
                    partial.resize(scratch_len, 0.0);
                }
                partial[..scratch_len].fill(0.0);
                partial.truncate(scratch_len);

                let parallel = Self::adaptive_parallel(route_count, self.component_out_dim, blocks_per_row);
                let compute_rows = |partial: &mut [f32]| {
                    partial
                        .par_chunks_mut(route_count)
                        .enumerate()
                        .for_each(|(row, accs)| {
                            if kernel >= 2 {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    Self::dot_streamed_routes_fused_avx2_fma(
                                        &x_data,
                                        &route_x_offsets,
                                        expert,
                                        row,
                                        self.in_dim,
                                        accs,
                                    );
                                    return;
                                }
                            } else if kernel == 1 {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    Self::dot_streamed_routes_fused_avx2(
                                        &x_data,
                                        &route_x_offsets,
                                        expert,
                                        row,
                                        self.in_dim,
                                        accs,
                                    );
                                    return;
                                }
                            }

                            let row_bytes =
                                blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                            let row_start = row * row_bytes;
                            for block_idx in 0..blocks_per_row {
                                let block_start =
                                    row_start + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                                let dequant =
                                    &MXFP4Layer::DEQUANT_LUT[expert[block_start] as usize];
                                let packed =
                                    &expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                                let col_start = block_idx * MXFP4_BLOCK_SIZE;
                                for (route_idx, &x_offset) in
                                    route_x_offsets.iter().enumerate()
                                {
                                    let x_row =
                                        &x_data[x_offset..x_offset + self.in_dim];
                                    for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                        let packed_byte = packed[byte_idx];
                                        accs[route_idx] += x_row[col_start + byte_idx]
                                            * dequant[(packed_byte & 0x0f) as usize];
                                        accs[route_idx] += x_row[
                                            col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx
                                        ] * dequant[(packed_byte >> 4) as usize];
                                    }
                                }
                            }
                        });
                };

                if parallel {
                    compute_rows(&mut partial[..scratch_len]);
                } else {
                    for row in 0..self.component_out_dim {
                        let accs =
                            &mut partial[row * route_count..(row + 1) * route_count];
                        if kernel >= 2 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                Self::dot_streamed_routes_fused_avx2_fma(
                                    &x_data,
                                    &route_x_offsets,
                                    expert,
                                    row,
                                    self.in_dim,
                                    accs,
                                );
                                continue;
                            }
                        } else if kernel == 1 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                Self::dot_streamed_routes_fused_avx2(
                                    &x_data,
                                    &route_x_offsets,
                                    expert,
                                    row,
                                    self.in_dim,
                                    accs,
                                );
                                continue;
                            }
                        }

                        let row_bytes =
                            blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);
                        for block_idx in 0..blocks_per_row {
                            let block_start =
                                row * row_bytes
                                    + block_idx * (MXFP4_BLOCK_SIZE / 2 + 1);
                            let dequant =
                                &MXFP4Layer::DEQUANT_LUT[expert[block_start] as usize];
                            let packed =
                                &expert[block_start + 1..block_start + 1 + MXFP4_BLOCK_SIZE / 2];
                            let col_start = block_idx * MXFP4_BLOCK_SIZE;
                            for (route_idx, &x_offset) in
                                route_x_offsets.iter().enumerate()
                            {
                                let x_row =
                                    &x_data[x_offset..x_offset + self.in_dim];
                                for byte_idx in 0..MXFP4_BLOCK_SIZE / 2 {
                                    let packed_byte = packed[byte_idx];
                                    accs[route_idx] += x_row[col_start + byte_idx]
                                        * dequant[(packed_byte & 0x0f) as usize];
                                    accs[route_idx] += x_row[
                                        col_start + MXFP4_BLOCK_SIZE / 2 + byte_idx
                                    ] * dequant[(packed_byte >> 4) as usize];
                                }
                            }
                        }
                    }
                }

                for (route_idx, &route_row) in routes.iter().enumerate() {
                    let out_row =
                        &mut output[route_row * self.out_dim..(route_row + 1) * self.out_dim];
                    let base = route_idx;
                    for row in 0..self.component_out_dim {
                        let col = if interleaved_gate_up {
                            row * 2 + component
                        } else {
                            row
                        };
                        out_row[col] += partial[row * route_count + base];
                    }
                }
            }
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn streaming_swiglu_avx2_matches_scalar() {
        let alpha = 1.702f32;
        let values = [
            -18.0f32, -8.0, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0,
            2.0, 4.0, 7.0, 18.0, -30.0, 30.0, 0.5, -0.5,
        ];
        let gates = values;
        let ups = [
            7.0f32, -7.0, 3.0, -3.0, 1.0, -1.0, 0.0, 2.0,
            -2.0, 6.0, -6.0, 7.0, -7.0, 0.0, 0.75, -0.75,
        ];

        let expected = gates.map(|gate| gate.min(7.0));
        let expected_up = ups.map(|up| up.clamp(-7.0, 7.0));
        let expected: Vec<f32> = expected
            .iter()
            .zip(expected_up.iter())
            .map(|(&gate, &up)| {
                (up + 1.0) * gate / (1.0 + (-gate * alpha).exp())
            })
            .collect();

        let actual = unsafe {
            let g = _mm256_loadu_ps(gates.as_ptr());
            let u = _mm256_loadu_ps(ups.as_ptr());
            let y = MxFp4StreamingExpertLayer::swiglu8_avx2(g, u, alpha);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), y);
            let g2 = _mm256_loadu_ps(gates.as_ptr().add(8));
            let u2 = _mm256_loadu_ps(ups.as_ptr().add(8));
            let y2 = MxFp4StreamingExpertLayer::swiglu8_avx2(g2, u2, alpha);
            _mm256_storeu_ps(out[8..].as_mut_ptr(), y2);
            out.to_vec()
        };

        for (got, want) in actual.iter().zip(expected.iter()) {
            let tol = 2e-4f32.max(want.abs() * 2e-4);
            assert!(
                (got - want).abs() <= tol,
                "got={got} want={want} diff={}",
                (got - want).abs()
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn streaming_fused_dequant_matches_reference_all_normal_scales() {
        let mut raw = vec![0u8; MXFP4_BLOCK_SIZE / 2 + 1];
        for scale in 2u16..=253 {
            raw[0] = scale as u8;
            for i in 0..MXFP4_BLOCK_SIZE / 2 {
                // Cover every nibble value repeatedly, including signed FP4 values.
                raw[i + 1] = (((i * 7) & 0x0f) as u8)
                    | ((((i * 11 + 3) & 0x0f) as u8) << 4);
            }

            let vectors = unsafe {
                MxFp4StreamingExpertLayer::load_fused_weight_vectors_avx2(&raw, 0)
            };
            let mut actual = [0.0f32; MXFP4_BLOCK_SIZE];
            unsafe {
                _mm256_storeu_ps(actual.as_mut_ptr(), vectors[0]);
                _mm256_storeu_ps(actual.as_mut_ptr().add(8), vectors[1]);
                _mm256_storeu_ps(actual.as_mut_ptr().add(16), vectors[2]);
                _mm256_storeu_ps(actual.as_mut_ptr().add(24), vectors[3]);
            }

            let dequant = &MXFP4Layer::DEQUANT_LUT[scale as usize];
            for i in 0..MXFP4_BLOCK_SIZE / 2 {
                let packed = raw[i + 1];
                let expected_lo = dequant[(packed & 0x0f) as usize];
                let expected_hi = dequant[(packed >> 4) as usize];
                assert_eq!(
                    actual[i].to_bits(),
                    expected_lo.to_bits(),
                    "scale={scale} lane={i} low"
                );
                assert_eq!(
                    actual[16 + i].to_bits(),
                    expected_hi.to_bits(),
                    "scale={scale} lane={} high",
                    16 + i
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn streaming_fused_row_kernel_matches_scalar() -> Result<()> {
        let in_dim = 64;
        let blocks_per_row = in_dim / MXFP4_BLOCK_SIZE;
        let row_bytes = blocks_per_row * (MXFP4_BLOCK_SIZE / 2 + 1);

        let x = (0..in_dim)
            .map(|i| ((i as f32 * 0.071) - 1.25).sin())
            .collect::<Vec<_>>();
        let mut raw = vec![0u8; row_bytes];
        for block in 0..blocks_per_row {
            let start = block * (MXFP4_BLOCK_SIZE / 2 + 1);
            raw[start] = if block == 0 { 120 } else { 132 };
            for i in 0..MXFP4_BLOCK_SIZE / 2 {
                raw[start + 1 + i] = (i as u8).wrapping_mul(29).wrapping_add(0x53);
            }
        }

        let mut expected = 0.0f32;
        for block in 0..blocks_per_row {
            let start = block * (MXFP4_BLOCK_SIZE / 2 + 1);
            let dequant = &MXFP4Layer::DEQUANT_LUT[raw[start] as usize];
            let col_start = block * MXFP4_BLOCK_SIZE;
            for i in 0..MXFP4_BLOCK_SIZE / 2 {
                let packed = raw[start + 1 + i];
                expected += x[col_start + i] * dequant[(packed & 0x0f) as usize];
                expected +=
                    x[col_start + MXFP4_BLOCK_SIZE / 2 + i]
                        * dequant[(packed >> 4) as usize];
            }
        }

        let kernel = MxFp4StreamingExpertLayer::dot_kernel();
        let actual = if kernel >= 2 {
            unsafe {
                MxFp4StreamingExpertLayer::dot_streamed_row_fused_avx2_fma(
                    &x,
                    &raw,
                    0,
                    in_dim,
                )
            }
        } else {
            unsafe {
                MxFp4StreamingExpertLayer::dot_streamed_row_fused_avx2(
                    &x,
                    &raw,
                    0,
                    in_dim,
                )
            }
        };

        assert!(
            (actual - expected).abs() < 1e-3,
            "actual={actual} expected={expected}"
        );
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
