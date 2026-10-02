use rayon::ThreadPool;
use std::sync::LazyLock;

pub(super) static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    let threads = std::env::var("MISTRALRS_ATTN_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(|| candle_core::utils::get_num_threads().max(1));
    let affinity = std::env::var("MISTRALRS_ATTN_AFFINITY")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "no"))
        .unwrap_or(true);

    let mut builder = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|idx| format!("mistralrs-attn-{idx}"));
    if affinity {
        builder = builder.start_handler(|_| candle_core::utils::set_thread_affinity());
    }
    builder
        .build()
        .expect("Failed to build custom Rayon thread-pool for flash-attention")
});
