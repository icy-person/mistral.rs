fn physical_core_count() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
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
            if let Ok(core_id) = fs::read_to_string(entry.path().join("topology/core_id")) {
                cores.insert(core_id.trim().to_string());
            }
        }
        return (!cores.is_empty()).then_some(cores.len());
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

use rayon::ThreadPool;
use std::sync::LazyLock;

pub(super) static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    let default_threads = candle_core::utils::get_num_threads().max(1);
    let threads = std::env::var("MISTRALRS_ATTN_THREADS")
        .ok()
        .and_then(|value| {
            if value.eq_ignore_ascii_case("physical") {
                physical_core_count().or(Some(default_threads))
            } else {
                value.parse::<usize>().ok().filter(|&value| value > 0)
            }
        })
        .unwrap_or(default_threads);
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
