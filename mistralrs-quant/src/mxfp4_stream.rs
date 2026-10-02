// The GGUF archive already owns read-only shard mmaps; streamed experts can
// reference those mappings directly without another mmap or read/copy.
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{self},
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
}

impl Default for MxFp4StreamConfig {
    fn default() -> Self {
        Self {
            cache_mb: None,
            cache_floor_mb: 1536,
            cache_ceil_mb: Some(4096),
            cache_per_source: 5,
            zero_copy: true,
            io_threads: 4,
            overlap: true,
            o_direct: false,
            stats: false,
        }
    }
}

impl MxFp4StreamConfig {
    pub(crate) fn from_env() -> Self {
        let defaults = Self::default();
        let cache_mb = std::env::var("MISTRALRS_MOE_CACHE_MB")
            .ok()
            .and_then(|v| parse_cache_mb(&v));
        let cache_floor_mb = env_usize(
            "MISTRALRS_MOE_CACHE_FLOOR_MB",
            defaults.cache_floor_mb,
        );
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
            .clamp(1, 32),
            zero_copy: env_bool("MISTRALRS_MOE_ZERO_COPY", defaults.zero_copy),
            io_threads: env_usize("MISTRALRS_MOE_IO_THREADS", defaults.io_threads).clamp(1, 32),
            overlap: env_bool("MISTRALRS_MOE_OVERLAP", defaults.overlap),
            o_direct: env_bool("MISTRALRS_MOE_O_DIRECT", defaults.o_direct),
            stats: env_bool("MISTRALRS_MOE_STATS", defaults.stats),
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
    pub source: String,
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
struct ReadJob {
    shard: usize,
    offset: u64,
    len: usize,
    file_len: usize,
    reply: SyncSender<io::Result<Arc<MxFp4StreamData>>>,
}

#[derive(Debug)]
pub(crate) enum MxFp4StreamData {
    Owned(Arc<[u8]>),
    Mapped {
        map: Arc<Mmap>,
        start: usize,
        len: usize,
    },
}

impl MxFp4StreamData {
    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::ArchiveMapped { len, .. } => *len,
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
    Pending(Receiver<io::Result<Arc<MxFp4StreamData>>>),
}

#[derive(Debug, Default)]
struct Stats {
    hits: AtomicU64,
    misses: AtomicU64,
    reads: AtomicU64,
    bytes_read: AtomicU64,
    evictions: AtomicU64,
    report_calls: AtomicU64,
}

#[derive(Debug)]
pub(crate) struct MxFp4StreamCache {
    inner: Mutex<CacheInner>,
    queues: Vec<SyncSender<ReadJob>>,
    next_queue: AtomicUsize,
    paths: Vec<PathBuf>,
    archive: Arc<crate::GgufArchive>,
    config: MxFp4StreamConfig,
    budget_bytes: usize,
    stats: Stats,
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

        let archive_for_cache = archive.clone();
        let mut queues = Vec::with_capacity(config.io_threads);
        let needs_io_workers = maps.iter().any(Option::is_none);

        if needs_io_workers {
            for worker_id in 0..config.io_threads {
                let (queue_tx, queue_rx) = mpsc::sync_channel::<ReadJob>(queue_size);
                queues.push(queue_tx);
                let worker_paths = paths.clone();
                let want_direct = config.o_direct;
                thread::Builder::new()
                    .name(format!("mxfp4-io-{worker_id}"))
                    .spawn(move || {
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
                            let result = read_file_range(
                                file,
                                direct_file,
                                None,
                                job.offset,
                                job.len,
                                job.file_len,
                            )
                            .map(Arc::new);
                            let _ = job.reply.send(result);
                        }
                    })
                    .map_err(|err| {
                        io::Error::new(
                            io::ErrorKind::Other,
                            format!("failed to start MXFP4 I/O worker: {err}"),
                        )
                    })?;
            }
        }


        let cache = Arc::new(Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                used_bytes: 0,
                clock: 0,
            }),
            queues,
            next_queue: AtomicUsize::new(0),
            paths: paths.clone(),
            archive: archive_for_cache,
            budget_bytes: config.cache_budget_bytes(),
            config,
            stats: Stats::default(),
        });
        Ok(cache)
    }

    pub(crate) fn overlap(&self) -> bool {
        self.config.overlap
    }

    fn lookup(&self, key: &MxFp4StreamKey) -> Option<Arc<MxFp4StreamData>> {
        let mut guard = self.inner.lock().ok()?;
        guard.clock = guard.clock.wrapping_add(1);
        let now = guard.clock;
        let entry = guard.entries.get_mut(&key)?;
        entry.last_used = now;
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        Some(entry.data.clone())
    }

    fn insert(&self, key: MxFp4StreamKey, data: Arc<MxFp4StreamData>) {
        let len = data.len();
        let capacity = self.budget_bytes;
        if len == 0 || len > capacity {
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

        // Preserve a small layer-local working set. A global LRU thrashes because
        // GPT-OSS walks layers serially while autoregressive decoding revisits the
        // same layer-specific experts on the next token. Keeping five recent experts
        // per tensor source matches the common top-4 routing footprint.
        let source = key.source.clone();
        while guard
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

        while guard.used_bytes.saturating_add(len) > capacity {
            let Some(victim) = guard
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(old) = guard.entries.remove(&victim) {
                guard.used_bytes = guard.used_bytes.saturating_sub(old.bytes);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        if guard.used_bytes.saturating_add(len) <= capacity {
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

    fn mapped_range(&self, range: MxFp4StreamRange) -> Option<Arc<MxFp4StreamData>> {
        if !self.config.zero_copy {
            return None;
        }
        let offset = usize::try_from(range.offset).ok()?;
        self.archive
            .shard_data_slice(range.shard, offset, range.len)
            .ok()?;
        Some(Arc::new(MxFp4StreamData::ArchiveMapped {
            archive: self.archive.clone(),
            shard: range.shard,
            offset,
            len: range.len,
        }))
    }


    fn submit(&self, range: MxFp4StreamRange) -> io::Result<Receiver<io::Result<Arc<MxFp4StreamData>>>> {
        if range.shard >= self.paths.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "MXFP4 shard index out of range",
            ));
        }
        let (tx, rx) = mpsc::sync_channel(1);
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        let worker = self.next_queue.fetch_add(1, Ordering::Relaxed) % self.queues.len().max(1);
        self.queues
            .get(worker)
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "MXFP4 I/O workers stopped"))?
            .send(ReadJob {
                shard: range.shard,
                offset: range.offset,
                len: range.len,
                file_len: range.file_len,
                reply: tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "MXFP4 I/O worker stopped"))?;
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
        self.stats.bytes_read.fetch_add(data.len() as u64, Ordering::Relaxed);
        let data = Arc::new(data);
        self.insert(key.clone(), data.clone());
        Ok(data)
    }

    pub(crate) fn prefetch(
        &self,
        requests: &[(MxFp4StreamKey, MxFp4StreamRange)],
    ) -> crate::Result<HashMap<MxFp4StreamKey, MxFp4StreamHandle>> {
        let mut result = HashMap::with_capacity(requests.len());
        for (key, range) in requests.iter().cloned() {
            if let Some(data) = self.lookup(&key) {
                result.insert(key, MxFp4StreamHandle::Ready(data));
            } else if let Some(data) = self.mapped_range(range) {
                self.insert(key.clone(), data.clone());
                result.insert(key, MxFp4StreamHandle::Ready(data));
            } else {
                result.insert(
                    key,
                    MxFp4StreamHandle::Pending(self.submit(range)?),
                );
            }
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
            MxFp4StreamHandle::Pending(rx) => {
                let data = rx
                    .recv()
                    .map_err(|_| candle_core::Error::Msg("MXFP4 I/O worker stopped".into()))?
                    .map_err(candle_core::Error::wrap)?;
                self.stats.reads.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_read.fetch_add(data.len() as u64, Ordering::Relaxed);
                self.insert(key.clone(), data.clone());
                Ok(data)
            }
        }
    }

    pub(crate) fn log_stats(&self) {
        if !self.config.stats {
            return;
        }
        let report = self.stats.report_calls.fetch_add(1, Ordering::Relaxed);
        if report != 0 && !report.is_multiple_of(32) {
            return;
        }
        let Ok(guard) = self.inner.lock() else {
            return;
        };
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let misses = self.stats.misses.load(Ordering::Relaxed);
        let reads = self.stats.reads.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_read.load(Ordering::Relaxed);
        let evictions = self.stats.evictions.load(Ordering::Relaxed);
        let requests = hits.saturating_add(misses);
        let hit_rate = if requests == 0 {
            0.0
        } else {
            hits as f64 / requests as f64
        };
        tracing::info!(
            target: "mistralrs_moe_stream",
            "GPT-OSS MXFP4 stream cache: entries={}, used_mib={}, budget_mib={}, per_source={}, hits={}, misses={}, hit_rate={:.1}%, reads={}, read_mib={}, evictions={}, io_threads={}, overlap={}, zero_copy={}, o_direct={}",
            guard.entries.len(),
            guard.used_bytes / MIB,
            self.budget_bytes / MIB,
            self.config.cache_per_source,
            hits,
            misses,
            hit_rate * 100.0,
            reads,
            bytes / MIB as u64,
            evictions,
            self.config.io_threads,
            self.config.overlap,
            self.config.zero_copy,
            self.config.o_direct,
        );
    }
}

fn read_file_range(
    file: &File,
    direct_file: Option<&File>,
    map: Option<&Arc<Mmap>>,
    offset: u64,
    len: usize,
    file_len: usize,
) -> io::Result<MxFp4StreamData> {
    let end = offset
        .checked_add(len as u64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "MXFP4 read range overflow"))?;
    if end > file_len as u64 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("MXFP4 read range {offset}..{end} exceeds file length {file_len}"),
        ));
    }

    if let Some(map) = map {
        let start = usize::try_from(offset).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "MXFP4 mmap offset exceeds usize")
        })?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "MXFP4 mmap range overflow"))?;
        if end > map.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "MXFP4 mmap range exceeds mapped file",
            ));
        }
        #[cfg(unix)]
        {
            let _ = map.advise_range(memmap2::Advice::WillNeed, start, len);
        }
        return Ok(MxFp4StreamData::Mapped {
            map: map.clone(),
            start,
            len,
        });
    }

    #[cfg(target_os = "linux")]
    if let Some(direct) = direct_file {
        let aligned_start = offset / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT;
        let aligned_end = end.div_ceil(DIRECT_ALIGNMENT as u64) * DIRECT_ALIGNMENT;
        if aligned_start < aligned_end && aligned_end <= file_len as u64 {
            let aligned_len = usize::try_from(aligned_end - aligned_start)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "aligned MXFP4 read too large"))?;
            match read_direct(direct, aligned_start, aligned_len) {
                Ok(buf) => {
                    let begin = usize::try_from(offset - aligned_start).unwrap_or(0);
                    return Ok(MxFp4StreamData::Owned(Arc::<[u8]>::from(
                        buf[begin..begin + len].to_vec(),
                    )));
                }
                Err(err) => {
                    tracing::debug!("O_DIRECT MXFP4 read fallback for range {offset}..{end}: {err}");
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
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "MXFP4 aligned allocation failed"));
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
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short MXFP4 expert read"));
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
            source: "blk.0.ffn_gate_exps.weight".to_string(),
            expert_index: 0,
        };
        let gate_l1 = MxFp4StreamKey {
            source: "blk.1.ffn_gate_exps.weight".to_string(),
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
            cache_per_source: 5,
            zero_copy: true,
            io_threads: 4,
            overlap: true,
            o_direct: false,
            stats: true,
        };
        assert_eq!(cfg.cache_budget_bytes(), 2048 * MIB);
    }
}
