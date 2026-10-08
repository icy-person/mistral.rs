use std::{
    sync::{atomic::AtomicUsize, Arc, OnceLock},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _CMP_GT_OQ, __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_and_si256,
    _mm256_castps256_ps128, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps,
    _mm256_cmpgt_epi32, _mm256_cmpeq_epi32, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32,
    _mm256_cvttps_epi32, _mm256_extractf128_ps, _mm256_fmadd_ps,
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
    zero_copy_experts: Vec<Vec<OnceLock<Arc<MxFp4StreamData>>>>,
    cache: Arc<MxFp4StreamCache>,
}

impl MxFp4StreamingExpertLayer {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn swiglu8_avx2(gate: __m256, up: __m256, alpha: f32) -> __m256 {
        swiglu8_avx2(gate, up, alpha)
    }

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
                self.cache.touch(key);
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
                        Self::physical_core_count().or(Some(default_threads))
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

/// Raw output pointer that may be shared across the MoE rayon pool.
///
/// Every parallel closure that uses it writes only to the column `row` it was
/// handed, so no two threads ever touch the same element. The accessor takes
/// `self` (not a field) so edition-2021 closures capture the whole wrapper
/// instead of the bare `*mut f32`, which is neither `Send` nor `Sync`.
#[derive(Clone, Copy)]
struct SharedOutPtr(*mut f32);

unsafe impl Send for SharedOutPtr {}
unsafe impl Sync for SharedOutPtr {}

impl SharedOutPtr {
    #[inline(always)]
    unsafe fn add(self, offset: usize) -> *mut f32 {
        self.0.add(offset)
    }
}

    /// Fused CPU GPT-OSS path for native streamed MXFP4 experts.
    ///
    /// Computes gate/up, GPT-OSS SwiGLU, down projection, and top-k weighted
    /// reduction without materializing Tensor intermediates.
    pub fn fused_gptoss_mlp(
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
        let mut activations: Vec<f32> = Vec::new();
        let kernel = MxFp4StreamingExpertLayer::dot_kernel();
        let approx_swiglu = approx_swiglu_enabled();
        let moe_threads = MxFp4StreamingExpertLayer::moe_thread_pool().current_num_threads();
        for &expert_idx in &experts {
            let (gate_data, down_data) = if let Some(queue) = pending.as_mut() {
                let gate_key = MxFp4StreamKey {
                    source: gate_up.cache_sources[0].clone(),
                    expert_index: expert_idx,
                };
                let gate_handle = queue.remove(&gate_key).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "fused GPT-OSS missing prefetched gate expert {expert_idx}"
                    ))
                })?;
                let gate_data = gate_up
                    .remember_zero_copy_expert(
                        0,
                        expert_idx,
                        gate_up.cache.resolve(&gate_key, gate_handle)?,
                    );

                let down_key = MxFp4StreamKey {
                    source: down.cache_sources[0].clone(),
                    expert_index: expert_idx,
                };
                let down_handle = queue.remove(&down_key).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "fused GPT-OSS missing prefetched down expert {expert_idx}"
                    ))
                })?;
                let down_data = down.remember_zero_copy_expert(
                    0,
                    expert_idx,
                    down.cache.resolve(&down_key, down_handle)?,
                );
                (gate_data, down_data)
            } else {
                let gate_key = MxFp4StreamKey {
                    source: gate_up.cache_sources[0].clone(),
                    expert_index: expert_idx,
                };
                let down_key = MxFp4StreamKey {
                    source: down.cache_sources[0].clone(),
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

                            gates[lane] = gate_bias_expert
                                .map(|bias| gate_value + bias[row * 2])
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
                let output_ptr = SharedOutPtr(output.as_mut_ptr());
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
                let output_ptr = SharedOutPtr(output.as_mut_ptr());
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