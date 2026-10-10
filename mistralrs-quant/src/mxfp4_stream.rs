use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io,
    ops::Deref,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender},
        Arc, Mutex,
    },
    thread,
};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(not(unix))]
use std::io::{Read, Seek, SeekFrom};
#[cfg(target_os = "linux")]
use std::{os::unix::fs::OpenOptionsExt, path::Path};

const MIB: usize = 1024 * 1024;
const DIRECT_ALIGNMENT: u64 = 4096;
const HOT_PROMOTION_HITS: u16 = 3;

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
        let host_available = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| parse_mem_available_bytes(&text));
        let cgroup_available = cgroup_available_memory_bytes();

        // In containers, host MemAvailable can be far larger than the memory
        // limit imposed on this process. The more restrictive amount is the
        // useful estimate for sizing an in-memory expert cache.
        min_available_memory_bytes(host_available, cgroup_available)
    }

    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

fn parse_mem_available_bytes(meminfo: &str) -> Option<usize> {
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|kib| kib.parse::<usize>().ok())
        .map(|kib| kib.saturating_mul(1024))
}

fn min_available_memory_bytes(
    host_available: Option<usize>,
    cgroup_available: Option<usize>,
) -> usize {
    match (host_available, cgroup_available) {
        (Some(host), Some(cgroup)) => host.min(cgroup),
        (Some(host), None) => host,
        (None, Some(cgroup)) => cgroup,
        (None, None) => 0,
    }
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
enum CgroupMemoryVersion {
    V1,
    V2,
}

#[cfg(target_os = "linux")]
fn cgroup_available_memory_bytes() -> Option<usize> {
    let cgroups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut available = None;

    for line in cgroups.lines() {
        let mut fields = line.splitn(3, ':');
        let Some(_hierarchy) = fields.next() else {
            continue;
        };
        let Some(controllers) = fields.next() else {
            continue;
        };
        let Some(group_path) = fields.next() else {
            continue;
        };

        let candidate = if controllers.is_empty() {
            cgroup_path_available_bytes(
                Path::new("/sys/fs/cgroup"),
                group_path,
                "memory.max",
                "memory.current",
                CgroupMemoryVersion::V2,
            )
        } else if controllers.split(',').any(|name| name == "memory") {
            cgroup_path_available_bytes(
                Path::new("/sys/fs/cgroup/memory"),
                group_path,
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                CgroupMemoryVersion::V1,
            )
        } else {
            None
        };

        if let Some(candidate) = candidate {
            available = Some(
                available.map_or(candidate, |current: usize| current.min(candidate)),
            );
        }
    }

    available
}

#[cfg(target_os = "linux")]
fn cgroup_path_available_bytes(
    mount: &Path,
    group_path: &str,
    limit_file: &str,
    usage_file: &str,
    version: CgroupMemoryVersion,
) -> Option<usize> {
    let mut current = mount.to_path_buf();
    for component in group_path.trim_start_matches('/').split('/') {
        if component.is_empty() {
            continue;
        }
        // Do not allow a malformed cgroup path to escape the known mount.
        if component == "." || component == ".." {
            return None;
        }
        current.push(component);
    }

    let mut available = None;
    loop {
        if let (Ok(limit), Ok(usage)) = (
            std::fs::read_to_string(current.join(limit_file)),
            std::fs::read_to_string(current.join(usage_file)),
        ) {
            let remaining = match version {
                CgroupMemoryVersion::V1 => parse_cgroup_v1_remaining_bytes(&limit, &usage),
                CgroupMemoryVersion::V2 => parse_cgroup_v2_remaining_bytes(&limit, &usage),
            };
            if let Some(remaining) = remaining {
                available = Some(
                    available.map_or(remaining, |current: usize| current.min(remaining)),
                );
            }
        }

        if current == mount || !current.pop() {
            break;
        }
    }

    available
}

fn parse_cgroup_v2_remaining_bytes(limit: &str, usage: &str) -> Option<usize> {
    let limit = limit.trim();
    if limit == "max" {
        return None;
    }

    let limit = limit.parse::<usize>().ok()?;
    let usage = usage.trim().parse::<usize>().ok()?;
    Some(limit.saturating_sub(usage))
}

fn parse_cgroup_v1_remaining_bytes(limit: &str, usage: &str) -> Option<usize> {
    let limit = limit.trim().parse::<u64>().ok()?;
    // cgroup v1 represents an unlimited memory limit with a very large
    // sentinel value (commonly just below i64::MAX).
    if limit >= (1_u64 << 60) {
        return None;
    }

    let usage = usage.trim().parse::<u64>().ok()?;
    usize::try_from(limit.saturating_sub(usage)).ok()
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
    promotions: AtomicU64,
    promoted_bytes: AtomicU64,
    advised_dontneed_bytes: AtomicU64,
    prefault_jobs: AtomicU64,
    prefault_nanos: AtomicU64,
    prefault_wait_nanos: AtomicU64,
    report_calls: AtomicU64,
    // Independent of logging: memory maintenance must still run when stats are disabled.
    maintenance_calls: AtomicU64,
}

#[derive(Debug)]
struct CacheEntry {
    data: Arc<MxFp4StreamData>,
    bytes: usize,
    last_used: u64,
    // Hot mapped ranges are copied into the bounded heap cache after repeated use.
    hits: u16,
    promotion_claimed: bool,
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
    queues: Vec<Sender<ReadJob>>,
    next_queue: AtomicUsize,
    prefault_queues: Vec<Sender<PrefaultJob>>,
    next_prefault_queue: AtomicUsize,
    paths: Vec<PathBuf>,
    archive: Arc<crate::GgufArchive>,
    config: MxFp4StreamConfig,
    budget_bytes: usize,
    stats: Arc<Stats>,
    warned_cache_cliff: std::sync::atomic::AtomicBool,
    // Process-wide storage bytes at cache construction; available on Linux via /proc/self/io.
    process_storage_read_bytes_at_start: Option<u64>,
}

impl MxFp4StreamCache {
    pub(crate) fn new(archive: &Arc<crate::GgufArchive>) -> io::Result<Arc<Self>> {
        let config = MxFp4StreamConfig::from_env();
        let paths = archive
            .shards()
            .iter()
            .map(|shard| shard.path().to_path_buf())
            .collect::<Vec<_>>();

        // These queues carry only per-expert job descriptors and one-shot reply
        // senders. Avoid blocking while scheduling a full routed-expert batch:
        // backpressure here can make prefetch wait for page faults or disk reads
        // to finish before the CPU has a chance to overlap them with compute.
        let mut queues = Vec::new();
        let mut prefault_queues = Vec::new();
        let stats = Arc::new(Stats::default());

        if config.zero_copy && config.overlap && config.prefault {
            let worker_count = config.io_threads.min(2).max(1);
            prefault_queues.reserve(worker_count);
            for worker_id in 0..worker_count {
                let (queue_tx, queue_rx) = mpsc::channel::<PrefaultJob>();
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
                let (queue_tx, queue_rx) = mpsc::channel::<ReadJob>();
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
            process_storage_read_bytes_at_start: process_storage_read_bytes(),
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
        // Share the hot-promotion path with the explicit fast-path touch. The
        // overlap/prefetch path uses lookup(), so promotion must happen here too.
        self.touch(key)
    }

    #[inline]
    pub(crate) fn touch(&self, key: &MxFp4StreamKey) -> Option<Arc<MxFp4StreamData>> {
        let (current, promote_range) = {
            let mut guard = self.inner.lock().ok()?;
            guard.clock = guard.clock.wrapping_add(1);
            let now = guard.clock;
            let entry = guard.entries.get_mut(key)?;
            entry.last_used = now;
            entry.hits = entry.hits.saturating_add(1);
            self.stats.hits.fetch_add(1, Ordering::Relaxed);

            let mapped_range = match entry.data.as_ref() {
                MxFp4StreamData::ArchiveMapped { shard, offset, len, .. }
                    if should_promote_mapped(entry.hits, *len, self.budget_bytes, entry.promotion_claimed) =>
                    Some((*shard, *offset, *len)),
                _ => None,
            };
            if mapped_range.is_some() {
                // Claim promotion while holding the cache lock to avoid duplicate copies.
                entry.promotion_claimed = true;
            }
            (entry.data.clone(), mapped_range)
        };

        if let Some((shard, offset, len)) = promote_range {
            if let Ok(bytes) = self.archive.shard_data_slice(shard, offset, len) {
                // Keep a bounded anonymous-RAM copy of experts that prove hot so
                // page-cache eviction doesn't force a later SSD read for the same expert.
                let owned = Arc::new(MxFp4StreamData::Owned(Arc::<[u8]>::from(bytes.to_vec())));
                self.insert(key.clone(), owned.clone());
                self.stats.promotions.fetch_add(1, Ordering::Relaxed);
                self.stats.promoted_bytes.fetch_add(len as u64, Ordering::Relaxed);
                return Some(owned);
            }
        }

        Some(current)
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
                    hits: 0,
                    promotion_claimed: false,
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

        // A rendezvous reply channel prevents prefault workers from touching
        // an unbounded number of mmap pages ahead of the consumer. The worker
        // can advance as soon as resolve() takes the current result.
        let (reply, rx) = mpsc::sync_channel(0);
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

        // Bound live read buffers by the number of I/O workers. With capacity
        // zero, a worker cannot read and retain every queued expert result while
        // the caller is still scheduling a layer's prefetch batch.
        let (tx, rx) = mpsc::sync_channel(0);
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
        if !self.zero_copy() && self.budget_bytes != 0 {
            let routed_bytes = requests
                .iter()
                .fold(0usize, |sum, (_, range)| sum.saturating_add(range.len));
            if routed_bytes > self.budget_bytes
                && !self.warned_cache_cliff.swap(true, Ordering::Relaxed)
            {
                tracing::warn!(
                    target: "mistralrs_moe_stream",
                    routed_working_set_mib = routed_bytes / MIB,
                    cache_budget_mib = self.budget_bytes / MIB,
                    "single-layer routed working set exceeds the heap cache budget; expect churn"
                );
            }
        }

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
        // Called from the routed forward path; cold-page maintenance must not
        // depend on the optional telemetry switch.
        let released_bytes = if self.config.release_cold && self.zero_copy() {
            let call = self.stats.maintenance_calls.fetch_add(1, Ordering::Relaxed).saturating_add(1);
            if release_cold_scan_due(call, true, true) { self.release_cold_pages() } else { 0 }
        } else { 0 };
        if released_bytes > 0 {
            self.stats.advised_dontneed_bytes.fetch_add(released_bytes as u64, Ordering::Relaxed);
        }

        if !self.config.stats { return; }

        let report = self.stats.report_calls.fetch_add(1, Ordering::Relaxed);
        if report != 0 && !report.is_multiple_of(256) { return; }

        let Ok(guard) = self.inner.lock() else { return; };
        let entry_count = guard.entries.len();
        let used_bytes = guard.used_bytes;
        let mapped_bytes = guard.entries.values().filter_map(|entry| match entry.data.as_ref() {
            MxFp4StreamData::ArchiveMapped { len, .. } => Some(*len),
            MxFp4StreamData::Owned(_) => None,
        }).sum::<usize>();
        drop(guard);

        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let reads = self.stats.reads.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_read.load(Ordering::Relaxed);
        let evictions = self.stats.evictions.load(Ordering::Relaxed);
        let promotions = self.stats.promotions.load(Ordering::Relaxed);
        let promoted_bytes = self.stats.promoted_bytes.load(Ordering::Relaxed);
        let advised_dontneed_bytes = self.stats.advised_dontneed_bytes.load(Ordering::Relaxed);
        let (mapped_resident_bytes, mapped_bytes_total) = self.mapped_residency();
        let mapped_residency = if mapped_bytes_total == 0 { 0.0 } else {
            mapped_resident_bytes as f64 / mapped_bytes_total as f64 * 100.0
        };

        // Unlike logical pread counters, /proc/self/io sees storage reads caused
        // by mmap page faults in Zero-Copy mode. This is process-wide, not model-exclusive.
        let process_storage_read_mib = self.process_storage_read_bytes_at_start
            .and_then(|start| process_storage_read_bytes().map(|current| current.saturating_sub(start)))
            .map(|value| value as f64 / MIB as f64)
            .unwrap_or(-1.0);

        let requests = hits.saturating_add(misses);
        let hit_rate = if requests == 0 { 0.0 } else { hits as f64 / requests as f64 };

        tracing::info!(
            target: "mistralrs_moe_stream",
            "GPT-OSS MXFP4 stream cache: entries={}, used_mib={}, budget_mib={}, mapped_mib={}, mapped_resident_mib={}, mapped_residency={:.1}%, per_source={}, hits={}, misses={}, hit_rate={:.1}%, reads={}, read_mib={}, process_storage_read_mib={:.1}, io_jobs={}, io_ms={}, io_wait_ms={}, prefault_jobs={}, prefault_ms={}, prefault_wait_ms={}, evictions={}, promotions={}, promoted_mib={}, advised_dontneed_total_mib={}, io_threads={}, overlap={}, zero_copy={}, o_direct={}, release_cold={}, release_idle={}, prefault={}",
            entry_count, used_bytes / MIB, self.budget_bytes / MIB, mapped_bytes / MIB,
            mapped_resident_bytes / MIB, mapped_residency, self.config.cache_per_source,
            hits, misses, hit_rate * 100.0, reads, bytes / (MIB as u64), process_storage_read_mib,
            self.stats.io_jobs.load(Ordering::Relaxed),
            self.stats.io_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.stats.io_wait_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.stats.prefault_jobs.load(Ordering::Relaxed),
            self.stats.prefault_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.stats.prefault_wait_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            evictions, promotions, promoted_bytes / MIB, advised_dontneed_bytes / (MIB as u64), self.config.io_threads, self.config.overlap,
            self.config.zero_copy, self.config.o_direct, self.config.release_cold,
            self.config.release_idle, self.config.prefault,
        );
    }
}

fn should_promote_mapped(hits: u16, len: usize, budget_bytes: usize, promotion_claimed: bool) -> bool {
    !promotion_claimed && hits >= HOT_PROMOTION_HITS && len > 0
        && budget_bytes > 0 && len <= budget_bytes
}

fn release_cold_scan_due(call: u64, release_cold: bool, zero_copy: bool) -> bool {
    release_cold && zero_copy && call > 0 && call.is_multiple_of(64)
}

fn parse_process_storage_read_bytes(io_text: &str) -> Option<u64> {
    io_text.lines()
        .find_map(|line| line.strip_prefix("read_bytes:"))
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn process_storage_read_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let io_text = std::fs::read_to_string("/proc/self/io").ok()?;
        parse_process_storage_read_bytes(&io_text)
    }
    #[cfg(not(target_os = "linux"))]
    { None }
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
    fn parses_mem_available_from_proc_meminfo() {
        let meminfo = "MemTotal: 8388608 kB\nMemAvailable: 4194304 kB\n";
        assert_eq!(
            parse_mem_available_bytes(meminfo),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(parse_mem_available_bytes("MemTotal: 1024 kB\n"), None);
    }

    #[test]
    fn cgroup_v2_budget_uses_remaining_memory() {
        assert_eq!(
            parse_cgroup_v2_remaining_bytes("8589934592\n", "6442450944\n"),
            Some(2 * 1024 * 1024 * 1024)
        );
        assert_eq!(parse_cgroup_v2_remaining_bytes("max\n", "1024\n"), None);
        assert_eq!(
            parse_cgroup_v2_remaining_bytes("1024\n", "2048\n"),
            Some(0)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_v2_budget_uses_tighter_ancestor_limit() {
        let root = std::env::temp_dir().join(format!(
            "mistralrs-cgroup-budget-{}",
            std::process::id()
        ));
        let child = root.join("child");
        std::fs::create_dir_all(&child).unwrap();

        std::fs::write(root.join("memory.max"), "1024\n").unwrap();
        std::fs::write(root.join("memory.current"), "800\n").unwrap();
        std::fs::write(child.join("memory.max"), "2048\n").unwrap();
        std::fs::write(child.join("memory.current"), "1024\n").unwrap();

        assert_eq!(
            cgroup_path_available_bytes(
                &root,
                "/child",
                "memory.max",
                "memory.current",
                CgroupMemoryVersion::V2,
            ),
            Some(224)
        );
        assert!(cgroup_path_available_bytes(
            &root,
            "../child",
            "memory.max",
            "memory.current",
            CgroupMemoryVersion::V2,
        )
        .is_none());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cgroup_v1_unlimited_sentinel_is_ignored() {
        assert_eq!(
            parse_cgroup_v1_remaining_bytes("9223372036854771712\n", "1024\n"),
            None
        );
        assert_eq!(
            parse_cgroup_v1_remaining_bytes("4096\n", "1024\n"),
            Some(3072)
        );
    }

    #[test]
    fn automatic_memory_estimate_respects_the_tighter_limit() {
        assert_eq!(
            min_available_memory_bytes(Some(12 * MIB), Some(2 * MIB)),
            2 * MIB
        );
        assert_eq!(min_available_memory_bytes(Some(12 * MIB), None), 12 * MIB);
        assert_eq!(min_available_memory_bytes(None, Some(2 * MIB)), 2 * MIB);
        assert_eq!(min_available_memory_bytes(None, None), 0);
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
    fn prefault_owned_buffer_is_safe() {
        let data = MxFp4StreamData::Owned(Arc::<[u8]>::from(vec![1u8; 8193]));
        assert!(prefault_mapped(&data).is_ok());
    }

    #[test]
    fn explicit_zero_source_quota_is_allowed() {
        let cfg = MxFp4StreamConfig {
            cache_mb: Some(1024),
            cache_floor_mb: 1536,
            cache_ceil_mb: Some(1024),
            cache_per_source: 0,
            zero_copy: false,
            io_threads: 1,
            overlap: false,
            o_direct: false,
            stats: false,
            release_cold: false,
            release_idle: 256,
            prefault: false,
        };
        assert_eq!(cfg.cache_per_source, 0);
        assert_eq!(cfg.cache_budget_bytes(), 1024 * MIB);
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

    #[test]
    fn cold_page_maintenance_is_rate_limited_and_not_logging_gated() {
        assert!(!release_cold_scan_due(63, true, true));
        assert!(release_cold_scan_due(64, true, true));
        assert!(!release_cold_scan_due(64, false, true));
        assert!(!release_cold_scan_due(64, true, false));
        assert!(!release_cold_scan_due(0, true, true));
    }

    #[test]
    fn parses_process_storage_read_counter() {
        let sample = "rchar: 1200\nwchar: 100\nsyscr: 12\nsyscw: 3\nread_bytes: 8192\nwrite_bytes: 4096\ncancelled_write_bytes: 0\n";
        assert_eq!(parse_process_storage_read_bytes(sample), Some(8192));
        assert_eq!(parse_process_storage_read_bytes("rchar: 1\n"), None);
    }

    #[test]
    fn promotes_only_repeated_mapped_experts_that_fit_the_budget() {
        assert!(!should_promote_mapped(2, 64, 1024, false));
        assert!(should_promote_mapped(3, 64, 1024, false));
        assert!(!should_promote_mapped(3, 64, 0, false));
        assert!(!should_promote_mapped(3, 2048, 1024, false));
        assert!(!should_promote_mapped(3, 64, 1024, true));
    }
}
