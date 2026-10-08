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
    released_bytes: AtomicU64,
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
    fn touch(&self, key: &MxFp4StreamKey) {
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

        let (tx, rx) = mpsc::sync_channel(1);
        let worker = self.next_queue.fetch_add(1, Ordering::Relaxed) % self.queues.len();

        self.queues[worker]
            .send(ReadJob {
                shard: range.shard,
                offset: range.offset,
                len: range.len,
                file_len: range.file_len,
                reply: tx,
            })
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "MXFP4 I/O worker stopped",
                )
            })?;

        Ok(rx)
    }

    pub(crate) fn load(
        &self,
        key: &MxFp4StreamKey,
        range: MxFp4StreamRange,
    ) -> crate::Result<Arc<MxFp4StreamData>> {
        if let Some(data) = self.lookup(key) {
            return Ok(data);
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);

        if let Some(data) = self.mapped_range(range) {
            self.insert(key.clone(), data.clone());
            return Ok(data);
        }

        let rx = self.submit(range)?;
        let data = rx
            .recv()
            .map_err(|_| candle_core::Error::Msg("MXFP4 I/O worker stopped".into()))?
            .map_err(candle_core::Error::wrap)?;
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.insert(key.clone(), data.clone());
        Ok(data)
    }

    pub(crate) fn prefetch(
        &self,
        requests: &[(MxFp4StreamKey, MxFp4StreamRange)],
    ) -> crate::Result<HashMap<MxFp4StreamKey, MxFp4StreamHandle>> {
        let mut result = HashMap::with_capacity(requests.len());

        for (key, range) in requests {
            if let Some(data) = self.lookup(key) {
                result.insert(key.clone(), MxFp4StreamHandle::Ready(data));
                continue;
            }

            self.stats.misses.fetch_add(1, Ordering::Relaxed);

            if let Some(data) = self.mapped_range(*range) {
                self.insert(key.clone(), data.clone());
                if let Some(rx) = self.submit_prefault(data.clone())? {
                    result.insert(key.clone(), MxFp4StreamHandle::PendingPrefault(rx));
                } else {
                    result.insert(key.clone(), MxFp4StreamHandle::Ready(data));
                }
                continue;
            }

            result.insert(key.clone(), MxFp4StreamHandle::PendingRead(self.submit(*range)?));
        }

        Ok(result)
    }

    pub(crate) fn resolve(
        &self,
        key: &MxFp4StreamKey,
        handle: MxFp4StreamHandle,
    ) -> crate::Result<Arc<MxFp4StreamData>> {
        match handle {
            MxFp4StreamHandle::Ready(data) => Ok(data),
            MxFp4StreamHandle::PendingRead(rx) => {
                let started = std::time::Instant::now();
                let data = rx
                    .recv()
                    .map_err(|_| candle_core::Error::Msg("MXFP4 I/O worker stopped".into()))?
                    .map_err(candle_core::Error::wrap)?;
                self.stats.io_wait_nanos.fetch_add(
                    started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    Ordering::Relaxed,
                );
                self.stats.reads.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_read
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                self.insert(key.clone(), data.clone());
                Ok(data)
            }
            MxFp4StreamHandle::PendingPrefault(rx) => {
                let started = std::time::Instant::now();
                let data = rx
                    .recv()
                    .map_err(|_| candle_core::Error::Msg("MXFP4 prefault worker stopped".into()))?
                    .map_err(candle_core::Error::wrap)?;
                self.stats.prefault_wait_nanos.fetch_add(
                    started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    Ordering::Relaxed,
                );
                self.insert(key.clone(), data.clone());
                Ok(data)
            }
        }
    }

    fn release_cold_pages(&self) -> usize {
        if !self.config.release_cold || !self.zero_copy() {
            return 0;
        }

        let Ok(guard) = self.inner.lock() else {
            return 0;
        };
        let now = guard.clock;
        let idle = self.config.release_idle;
        let candidates = guard
            .entries
            .values()
            .filter_map(|entry| {
                if now.wrapping_sub(entry.last_used) < idle {
                    return None;
                }
                match entry.data.as_ref() {
                    MxFp4StreamData::ArchiveMapped {
                        shard,
                        offset,
                        len,
                        ..
                    } if *len != 0 => Some((*shard, *offset, *len)),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        drop(guard);

        let mut released = 0usize;
        for (shard, offset, len) in candidates {
            if self.archive.shard_data_dont_need(shard, offset, len).is_ok() {
                released = released.saturating_add(len);
            }
        }
        if released != 0 {
            self.stats
                .released_bytes
                .fetch_add(released as u64, Ordering::Relaxed);
        }
        released
    }

    fn mapped_residency(&self) -> (usize, usize) {
        let ranges = {
            let Ok(guard) = self.inner.lock() else {
                return (0, 0);
            };
            guard
                .entries
                .values()
                .filter_map(|entry| match entry.data.as_ref() {
                    MxFp4StreamData::ArchiveMapped {
                        shard,
                        offset,
                        len,
                        ..
                    } => Some((*shard, *offset, *len)),
                    MxFp4StreamData::Owned(_) => None,
                })
                .collect::<Vec<_>>()
        };

        let mut resident_bytes = 0usize;
        let mut total_bytes = 0usize;
        for (shard, offset, len) in ranges {
            if let Ok((resident, total)) =
                self.archive.shard_data_residency(shard, offset, len)
            {
                resident_bytes = resident_bytes.saturating_add(resident);
                total_bytes = total_bytes.saturating_add(total);
            }
        }
        (resident_bytes, total_bytes)
    }

    pub(crate) fn log_stats(&self) {
        if !self.config.stats {
            return;
        }

        let report = self.stats.report_calls.fetch_add(1, Ordering::Relaxed);
        if report != 0 && !report.is_multiple_of(256) {
            return;
        }

        let Ok(guard) = self.inner.lock() else {
            return;
        };
        let entry_count = guard.entries.len();
        let used_bytes = guard.used_bytes;
        let mapped_bytes = guard
            .entries
            .values()
            .filter_map(|entry| match entry.data.as_ref() {
                MxFp4StreamData::ArchiveMapped { len, .. } => Some(*len),
                MxFp4StreamData::Owned(_) => None,
            })
            .sum::<usize>();
        drop(guard);

        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let reads = self.stats.reads.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_read.load(Ordering::Relaxed);
        let evictions = self.stats.evictions.load(Ordering::Relaxed);
        let (mapped_resident_bytes, mapped_bytes_total) = self.mapped_residency();
        let released_bytes = self.release_cold_pages();
        let mapped_residency = if mapped_bytes_total == 0 {
            0.0
        } else {
            mapped_resident_bytes as f64 / mapped_bytes_total as f64 * 100.0
        };

        let requests = hits.saturating_add(misses);
        let hit_rate = if requests == 0 {
            0.0
        } else {
            hits as f64 / requests as f64
        };

        tracing::info!(
            target: "mistralrs_moe_stream",
            "GPT-OSS MXFP4 stream cache: entries={}, used_mib={}, budget_mib={}, mapped_mib={}, mapped_resident_mib={}, mapped_residency={:.1}%, per_source={}, hits={}, misses={}, hit_rate={:.1}%, reads={}, read_mib={}, io_jobs={}, io_ms={}, io_wait_ms={}, prefault_jobs={}, prefault_ms={}, prefault_wait_ms={}, evictions={}, released_mib={}, io_threads={}, overlap={}, zero_copy={}, o_direct={}, release_cold={}, release_idle={}, prefault={}",
            entry_count,
            used_bytes / MIB,
            self.budget_bytes / MIB,
            mapped_bytes / MIB,
            mapped_resident_bytes / MIB,
            mapped_residency,
            self.config.cache_per_source,
            hits,
            misses,
            hit_rate * 100.0,
            reads,
            bytes / (MIB as u64),
            self.stats.io_jobs.load(Ordering::Relaxed),
            self.stats.io_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.stats.io_wait_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.stats.prefault_jobs.load(Ordering::Relaxed),
            self.stats.prefault_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.stats.prefault_wait_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            evictions,
            released_bytes / MIB,
            self.config.io_threads,
            self.config.overlap,
            self.config.zero_copy,
            self.config.o_direct,
            self.config.release_cold,
            self.config.release_idle,
            self.config.prefault,
        );
    }
}

fn prefault_mapped(data: &MxFp4StreamData) -> io::Result<()> {
    let bytes = data.deref();
    if bytes.is_empty() {
        return Ok(());
    }

    #[cfg(unix)]
    let page_size = {
        let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if value <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "unable to determine system page size for MXFP4 prefault",
            ));
        }
        value as usize
    };

    #[cfg(not(unix))]
    let page_size = 4096usize;

    let mut checksum = 0u8;
    let mut offset = 0usize;
    while offset < bytes.len() {
        // SAFETY: offset is inside the validated mapping or owned buffer.
        // This deliberately faults one byte per page without retaining another copy.
        checksum ^= unsafe { std::ptr::read_volatile(bytes.as_ptr().add(offset)) };
        offset = offset.saturating_add(page_size);
    }
    checksum ^= bytes[bytes.len() - 1];
    std::hint::black_box(checksum);
    Ok(())
}

fn read_file_range(
    file: &File,
    direct_file: Option<&File>,
    offset: u64,
    len: usize,
    file_len: usize,
) -> io::Result<MxFp4StreamData> {
    let end = offset
        .checked_add(len as u64)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "MXFP4 read range overflow",
            )
        })?;

    if end > file_len as u64 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("MXFP4 read range {offset}..{end} exceeds file length {file_len}"),
        ));
    }

    #[cfg(target_os = "linux")]
    if let Some(direct) = direct_file {
        let aligned_start = offset / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT;
        let aligned_end = end.div_ceil(DIRECT_ALIGNMENT) * DIRECT_ALIGNMENT;

        if aligned_start < aligned_end && aligned_end <= file_len as u64 {
            let aligned_len = usize::try_from(aligned_end - aligned_start).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "aligned MXFP4 read too large",
                )
            })?;

            match read_direct(direct, aligned_start, aligned_len) {
                Ok(buf) => {
                    let begin = usize::try_from(offset - aligned_start).unwrap_or(0);
                    return Ok(MxFp4StreamData::Owned(Arc::<[u8]>::from(
                        buf[begin..begin + len].to_vec(),
                    )));
                }
                Err(err) => {
                    tracing::debug!(
                        "O_DIRECT MXFP4 read fallback for range {offset}..{end}: {err}"
                    );
                }
            }
        }
    }

    let mut buf = vec![0u8; len];
    read_exact_at(file, offset, &mut buf)?;
    Ok(MxFp4StreamData::Owned(Arc::<[u8]>::from(buf)))
}

#[cfg(target_os = "linux")]
fn read_direct(file: &File, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let mut aligned = AlignedBuffer::new(len)?;
    read_exact_at(file, offset, aligned.as_mut_slice())?;
    Ok(aligned.to_vec())
}

#[cfg(target_os = "linux")]
struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
}

#[cfg(target_os = "linux")]
impl AlignedBuffer {
    fn new(len: usize) -> io::Result<Self> {
        let layout = std::alloc::Layout::from_size_align(len, DIRECT_ALIGNMENT as usize)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "MXFP4 aligned allocation failed",
            ));
        }

        Ok(Self { ptr, len, layout })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn to_vec(&self) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len).to_vec() }
    }
}

#[cfg(target_os = "linux")]
impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

fn read_exact_at(file: &File, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut done = 0usize;

    while done < buf.len() {
        #[cfg(unix)]
        let n = FileExt::read_at(file, &mut buf[done..], offset)?;

        #[cfg(not(unix))]
        let n = {
            let mut clone = file.try_clone()?;
            clone.seek(SeekFrom::Start(offset))?;
            clone.read(&mut buf[done..])?
        };

        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short MXFP4 expert read",
            ));
        }

        done += n;
        offset += n as u64;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cache_modes() {
        assert_eq!(parse_cache_mb("auto"), None);
        assert_eq!(parse_cache_mb("1536"), Some(1536));
        assert_eq!(parse_cache_mb("bogus"), None);
    }

    #[test]
    fn cache_keys_include_tensor_source() {
        let gate_l0 = MxFp4StreamKey {
            source: Arc::<str>::from("blk.0.ffn_gate_exps.weight"),
            expert_index: 0,
        };
        let gate_l1 = MxFp4StreamKey {
            source: Arc::<str>::from("blk.1.ffn_gate_exps.weight"),
            expert_index: 0,
        };
        assert_ne!(gate_l0, gate_l1);
    }

    #[test]
    fn explicit_cache_budget_is_capped() {
        let cfg = MxFp4StreamConfig {
            cache_mb: Some(4096),
            cache_floor_mb: 1536,
            cache_ceil_mb: Some(2048),
            cache_per_source: 0,
            zero_copy: true,
            io_threads: 4,
            overlap: true,
            o_direct: false,
            stats: true,
            release_cold: false,
            release_idle: 4096,
            prefault: true,
        };
        assert_eq!(cfg.cache_budget_bytes(), 2048 * MIB);
    }
}
