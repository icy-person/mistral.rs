use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io,
    ops::Deref,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender},
        Arc, Mutex,
    },
    thread,
};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(not(unix))]
use std::io::{Read, Seek, SeekFrom};
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

const MIB: usize = 1024 * 1024;
const DIRECT_ALIGNMENT: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MxFp4StreamConfig {
    pub cache_mb: Option<usize>,
    pub cache_floor_mb: usize,
    pub cache_ceil_mb: Option<usize>,
    pub cache_per_source: usize,
    pub zero_copy: bool,
    pub io_threads: usize,
    pub overlap: bool,
    pub o_direct: bool,
    pub stats: bool,
    pub release_cold: bool,
    pub release_idle: u64,
    pub prefault: bool,
}

impl Default for MxFp4StreamConfig {
    fn default() -> Self {
        Self {
            cache_mb: None,
            cache_floor_mb: 1536,
            cache_ceil_mb: Some(4096),
            cache_per_source: 0,
            zero_copy: true,
            io_threads: 4,
            overlap: true,
            o_direct: false,
            stats: false,
            release_cold: false,
            release_idle: 4096,
            prefault: true,
        }
    }
}

impl MxFp4StreamConfig {
    pub(crate) fn from_env() -> Self {
        let defaults = Self::default();
        let cache_mb = std::env::var("MISTRALRS_MOE_CACHE_MB")
            .ok()
            .and_then(|v| parse_cache_mb(&v));
        let cache_floor_mb =
            env_usize("MISTRALRS_MOE_CACHE_FLOOR_MB", defaults.cache_floor_mb);
        let cache_ceil_mb = match std::env::var("MISTRALRS_MOE_CACHE_CEIL_MB") {
            Ok(v) if v.trim().eq_ignore_ascii_case("none") => None,
            Ok(v) => v.parse::<usize>().ok().or(defaults.cache_ceil_mb),
            Err(_) => defaults.cache_ceil_mb,
        };

        Self {
            cache_mb,
            cache_floor_mb,
            cache_ceil_mb,
            cache_per_source: env_usize(
                "MISTRALRS_MOE_CACHE_PER_SOURCE",
                defaults.cache_per_source,
            )
            .min(128),
            zero_copy: env_bool("MISTRALRS_MOE_ZERO_COPY", defaults.zero_copy),
            io_threads: env_usize("MISTRALRS_MOE_IO_THREADS", defaults.io_threads).clamp(1, 32),
            overlap: env_bool("MISTRALRS_MOE_OVERLAP", defaults.overlap),
            o_direct: env_bool("MISTRALRS_MOE_O_DIRECT", defaults.o_direct),
            stats: env_bool("MISTRALRS_MOE_STATS", defaults.stats),
            release_cold: env_bool("MISTRALRS_MOE_RELEASE_COLD", defaults.release_cold),
            release_idle: env_u64("MISTRALRS_MOE_RELEASE_IDLE", defaults.release_idle).max(256),
            prefault: env_bool("MISTRALRS_MOE_PREFAULT", defaults.prefault),
        }
    }

    pub(crate) fn cache_budget_bytes(self) -> usize {
        let mb = match self.cache_mb {
            Some(mb) => mb,
            None => available_memory_bytes()
                .saturating_sub(self.cache_floor_mb.saturating_mul(MIB))
                / MIB,
        };
        self.cache_ceil_mb
            .map_or(mb, |cap| mb.min(cap))
            .saturating_mul(MIB)
    }
}

fn parse_cache_mb(value: &str) -> Option<usize> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("auto") {
        None
    } else {
        value.parse::<usize>().ok()
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn available_memory_bytes() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            if let Some(kib) = text
                .lines()
                .find_map(|line| line.strip_prefix("MemAvailable:"))
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse::<usize>().ok())
            {
                return kib.saturating_mul(1024);
            }
        }
    }
    0
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MxFp4StreamKey {
    pub source: Arc<str>,
    pub expert_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MxFp4StreamRange {
    pub shard: usize,
    pub offset: u64,
    pub len: usize,
    pub file_len: usize,
}

#[derive(Debug)]
pub(crate) enum MxFp4StreamData {
    Owned(Arc<[u8]>),
    ArchiveMapped {
        archive: Arc<crate::GgufArchive>,
        shard: usize,
        offset: usize,
        len: usize,
    },
}

impl MxFp4StreamData {
    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::ArchiveMapped { .. } => 0,
        }
    }
}

impl Deref for MxFp4StreamData {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(data) => data,
            Self::ArchiveMapped {
                archive,
                shard,
                offset,
                len,
            } => archive
                .shard_data_slice(*shard, *offset, *len)
                .expect("validated zero-copy GGUF MXFP4 range"),
        }
    }
}

#[derive(Debug)]
struct ReadJob {
    shard: usize,
    offset: u64,
    len: usize,
    file_len: usize,
    reply: SyncSender<io::Result<Arc<MxFp4StreamData>>>,
}

struct PrefaultJob {
    data: Arc<MxFp4StreamData>,
    reply: SyncSender<io::Result<Arc<MxFp4StreamData>>>,
}

struct WorkerFiles {
    normal: Vec<File>,
    direct: Vec<Option<File>>,
}

impl WorkerFiles {
    fn new(paths: &[PathBuf], want_direct: bool) -> io::Result<Self> {
        let mut normal = Vec::with_capacity(paths.len());
        let mut direct = Vec::with_capacity(paths.len());

        for path in paths {
            normal.push(File::open(path)?);

            #[cfg(target_os = "linux")]
            let direct_file = if want_direct {
                match OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)
                {
                    Ok(file) => Some(file),
                    Err(err) => {
                        tracing::debug!(
                            "O_DIRECT unavailable for {}: {err}",
                            path.display()
                        );
                        None
                    }
                }
            } else {
                None
            };

            #[cfg(not(target_os = "linux"))]
            let direct_file = None;

            direct.push(direct_file);
        }

        Ok(Self { normal, direct })
    }
}

pub(crate) enum MxFp4StreamHandle {
    Ready(Arc<MxFp4StreamData>),
    PendingRead(Receiver<io::Result<Arc<MxFp4StreamData>>>),
    PendingPrefault(Receiver<io::Result<Arc<MxFp4StreamData>>>),
}

#[derive(Debug, Default)]
struct Stats {
    hits: AtomicU64,
    misses: AtomicU64,
    reads: AtomicU64,
    bytes_read: AtomicU64,
    io_jobs: AtomicU64,
    io_nanos: AtomicU64,
    io_wait_nanos: AtomicU64,
    evictions: AtomicU64,
    prefault_jobs: AtomicU64,
    prefault_nanos: AtomicU64,
    prefault_wait_nanos: AtomicU64,
    report_calls: AtomicU64,
}

#[derive(Debug)]
struct CacheEntry {
    data: Arc<MxFp4StreamData>,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug)]
struct CacheInner {
    entries: HashMap<MxFp4StreamKey, CacheEntry>,
    used_bytes: usize,
    clock: u64,
}

#[derive(Debug)]
pub(crate) struct MxFp4StreamCache {
    inner: Mutex<CacheInner>,
    queues: Vec<SyncSender<ReadJob>>,
    next_queue: AtomicUsize,
    prefault_queues: Vec<SyncSender<PrefaultJob>>,
    next_prefault_queue: AtomicUsize,
    paths: Vec<PathBuf>,
    archive: Arc<crate::GgufArchive>,
    config: MxFp4StreamConfig,
    budget_bytes: usize,
    stats: Arc<Stats>,
    warned_cache_cliff: std::sync::atomic::AtomicBool,
}

impl MxFp4StreamCache {
    pub(crate) fn new(archive: &Arc<crate::GgufArchive>) -> io::Result<Arc<Self>> {
        let config = MxFp4StreamConfig::from_env();
        let paths = archive
            .shards()
            .iter()
            .map(|shard| shard.path().to_path_buf())
            .collect::<Vec<_>>();

        let queue_size = 8usize;
        let mut queues = Vec::new();
        let mut prefault_queues = Vec::new();
        let stats = Arc::new(Stats::default());

        if config.zero_copy && config.overlap && config.prefault {
            let worker_count = config.io_threads.min(2).max(1);
            prefault_queues.reserve(worker_count);
            for worker_id in 0..worker_count {
                let (queue_tx, queue_rx) = mpsc::sync_channel::<PrefaultJob>(queue_size);
                prefault_queues.push(queue_tx);
                let worker_stats = stats.clone();
                thread::Builder::new()
                    .name(format!("mxfp4-prefault-{worker_id}"))
                    .spawn(move || {
                        while let Ok(job) = queue_rx.recv() {
                            let started = std::time::Instant::now();
                            let result = prefault_mapped(&job.data).map(|_| job.data.clone());
                            worker_stats.prefault_jobs.fetch_add(1, Ordering::Relaxed);
                            worker_stats.prefault_nanos.fetch_add(
                                started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                                Ordering::Relaxed,
                            );
                            let _ = job.reply.send(result);
                        }
                    })
                    .map_err(|err| {
                        io::Error::new(
                            io::ErrorKind::Other,
                            format!("failed to start MXFP4 prefault worker: {err}"),
                        )
                    })?;
            }
        }

        if !config.zero_copy || config.o_direct {
            queues.reserve(config.io_threads);
            for worker_id in 0..config.io_threads {
                let (queue_tx, queue_rx) = mpsc::sync_channel::<ReadJob>(queue_size);
                queues.push(queue_tx);

                let worker_paths = paths.clone();
                let want_direct = config.o_direct;
                thread::Builder::new()
                    .name(format!("mxfp4-io-{worker_id}"))
                    .spawn({
                        let worker_stats = stats.clone();
                        move || {
                        let files = match WorkerFiles::new(&worker_paths, want_direct) {
                            Ok(files) => files,
                            Err(err) => {
                                tracing::error!(
                                    "failed to open GPT-OSS MXFP4 shard files for worker {worker_id}: {err}"
                                );
                                return;
                            }
                        };

                        while let Ok(job) = queue_rx.recv() {
                            let Some(file) = files.normal.get(job.shard) else {
                                let _ = job.reply.send(Err(io::Error::new(
                                    io::ErrorKind::InvalidInput,
                                    "MXFP4 shard index out of range",
                                )));
                                continue;
                            };

                            let direct_file =
                                files.direct.get(job.shard).and_then(Option::as_ref);
                            let started = std::time::Instant::now();
                            let result = read_file_range(
                                file,
                                direct_file,
                                job.offset,
                                job.len,
                                job.file_len,
                            )
                            .map(Arc::new);
                            worker_stats.io_jobs.fetch_add(1, Ordering::Relaxed);
                            worker_stats.io_nanos.fetch_add(
                                started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                                Ordering::Relaxed,
                            );
                            let _ = job.reply.send(result);
                        }
                    }})
                    .map_err(|err| {
                        io::Error::new(
                            io::ErrorKind::Other,
                            format!("failed to start MXFP4 I/O worker: {err}"),
                        )
                    })?;
            }
        }

        Ok(Arc::new(Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                used_bytes: 0,
                clock: 0,
            }),
            queues,
            next_queue: AtomicUsize::new(0),
            prefault_queues,
            next_prefault_queue: AtomicUsize::new(0),
            paths,
            archive: archive.clone(),
            budget_bytes: config.cache_budget_bytes(),
            config,
            stats,
            warned_cache_cliff: std::sync::atomic::AtomicBool::new(false),
        }))
    }

    #[inline(always)]
    pub(crate) fn overlap(&self) -> bool {
        self.config.overlap
    }

    #[inline(always)]
    pub(crate) fn zero_copy(&self) -> bool {
        self.config.zero_copy && !self.config.o_direct
    }

    #[inline]
    fn lookup(&self, key: &MxFp4StreamKey) -> Option<Arc<MxFp4StreamData>> {
        let mut guard = self.inner.lock().ok()?;
        guard.clock = guard.clock.wrapping_add(1);
        let now = guard.clock;
        let entry = guard.entries.get_mut(key)?;
        entry.last_used = now;
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        Some(entry.data.clone())
    }

    #[inline]
    pub(crate) fn touch(&self, key: &MxFp4StreamKey) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        guard.clock = guard.clock.wrapping_add(1);
        let now = guard.clock;
        if let Some(entry) = guard.entries.get_mut(key) {
            entry.last_used = now;
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn insert(&self, key: MxFp4StreamKey, data: Arc<MxFp4StreamData>) {
        let len = data.len();
        if len > self.budget_bytes {
            return;
        }

        let Ok(mut guard) = self.inner.lock() else {
            return;
        };

        guard.clock = guard.clock.wrapping_add(1);
        let now = guard.clock;

        if let Some(old) = guard.entries.remove(&key) {
            guard.used_bytes = guard.used_bytes.saturating_sub(old.bytes);
        }

        // ArchiveMapped entries are just tiny descriptors pointing at the
        // existing GGUF mmap, so they consume no heap-cache budget. Do not
        // evict them by the per-source quota: keeping the full routed-expert
        // working set avoids repeated cache misses and repeated page advice.
        let heap_backed = len != 0;

        if heap_backed {
            let source = key.source.clone();
            while self.config.cache_per_source > 0
                && guard
                    .entries
                    .iter()
                    .filter(|(entry_key, _)| entry_key.source == source)
                    .count()
                    >= self.config.cache_per_source
            {
                let Some(victim) = guard
                    .entries
                    .iter()
                    .filter(|(entry_key, _)| entry_key.source == source)
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(entry_key, _)| entry_key.clone())
                else {
                    break;
                };

                if let Some(old) = guard.entries.remove(&victim) {
                    guard.used_bytes = guard.used_bytes.saturating_sub(old.bytes);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }

            while guard.used_bytes.saturating_add(len) > self.budget_bytes {
                let Some(victim) = guard
                    .entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(entry_key, _)| entry_key.clone())
                else {
                    break;
                };

                if let Some(old) = guard.entries.remove(&victim) {
                    guard.used_bytes = guard.used_bytes.saturating_sub(old.bytes);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Zero-copy mappings are not constrained by the heap byte budget.
        if !heap_backed || guard.used_bytes.saturating_add(len) <= self.budget_bytes {
            guard.used_bytes = guard.used_bytes.saturating_add(len);
            guard.entries.insert(
                key,
                CacheEntry {
                    data,
                    bytes: len,
                    last_used: now,
                },
            );
        }
    }

    #[inline]
    fn mapped_range(&self, range: MxFp4StreamRange) -> Option<Arc<MxFp4StreamData>> {
        if !self.config.zero_copy || self.config.o_direct {
            return None;
        }

        let offset = usize::try_from(range.offset).ok()?;
        self.archive
            .shard_data_slice(range.shard, offset, range.len)
            .ok()?;
        let _ = self
            .archive
            .shard_data_will_need(range.shard, offset, range.len);

        Some(Arc::new(MxFp4StreamData::ArchiveMapped {
            archive: self.archive.clone(),
            shard: range.shard,
            offset,
            len: range.len,
        }))
    }

    fn submit_prefault(
        &self,
        data: Arc<MxFp4StreamData>,
    ) -> io::Result<Option<Receiver<io::Result<Arc<MxFp4StreamData>>>>> {
        if self.prefault_queues.is_empty() {
            return Ok(None);
        }

        let (reply, rx) = mpsc::sync_channel(1);
        let worker = self
            .next_prefault_queue
            .fetch_add(1, Ordering::Relaxed)
            % self.prefault_queues.len();

        self.prefault_queues[worker]
            .send(PrefaultJob { data, reply })
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "MXFP4 prefault worker stopped",
                )
            })?;

        Ok(Some(rx))
    }

    fn submit(
        &self,
        range: MxFp4StreamRange,
    ) -> io::Result<Receiver<io::Result<Arc<MxFp4StreamData>>>> {
        if self.queues.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "MXFP4 zero-copy mapping unavailable and no I/O workers are configured",
            ));
        }
        if range.shard >= self.paths.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "MXFP4 shard index out of range",
            ));
        }
