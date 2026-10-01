//! BigMoeOnEdge-style routed expert streaming for GGUF MoE models.
//!
//! The GGUF archive is used as an index. Routed expert matrices are loaded by exact byte range,
//! optionally through O_DIRECT, converted to the existing quantized linear implementations, and
//! retained in a shared bounded LRU cache. When overlap is enabled, all unique experts needed by a
//! projection are queued before the first expert is computed, so storage I/O runs concurrently
//! with CPU compute.

use candle_core::{quantized::GgmlDType, Device, DType, Result, Tensor};
use mistralrs_quant::{GgufArchive, GgufMatMul, MXFP4Layer, QuantMethod};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender},
        Arc, Mutex, OnceLock, Weak,
    },
    thread,
};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

const MIB: usize = 1024 * 1024;
const DIRECT_ALIGNMENT: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MoeStreamConfig {
    pub enabled: bool,
    pub cache_mb: Option<usize>,
    pub cache_floor_mb: usize,
    pub cache_ceil_mb: Option<usize>,
    pub io_threads: usize,
    pub overlap: bool,
    pub o_direct: bool,
    pub stats: bool,
}

impl Default for MoeStreamConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cache_mb: None,
            cache_floor_mb: 1536,
            cache_ceil_mb: Some(4096),
            io_threads: 4,
            overlap: true,
            o_direct: false,
            stats: false,
        }
    }
}

impl MoeStreamConfig {
    pub(super) fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            enabled: env_bool("MISTRALRS_MOE_STREAM", false),
            cache_mb: parse_cache_mb(
                &std::env::var("MISTRALRS_MOE_CACHE_MB")
                    .unwrap_or_else(|_| "auto".to_string()),
            ),
            cache_floor_mb: env_usize(
                "MISTRALRS_MOE_CACHE_FLOOR_MB",
                defaults.cache_floor_mb,
            ),
            cache_ceil_mb: std::env::var("MISTRALRS_MOE_CACHE_CEIL_MB")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .or(defaults.cache_ceil_mb),
            io_threads: env_usize("MISTRALRS_MOE_IO_THREADS", defaults.io_threads).clamp(1, 32),
            overlap: env_bool("MISTRALRS_MOE_OVERLAP", defaults.overlap),
            o_direct: env_bool("MISTRALRS_MOE_O_DIRECT", defaults.o_direct),
            stats: env_bool("MISTRALRS_MOE_STATS", defaults.stats),
        }
    }

    fn cache_budget_bytes(self) -> usize {
        let mb = match self.cache_mb {
            Some(mb) => mb,
            None => available_memory_bytes()
                .saturating_sub(self.cache_floor_mb.saturating_mul(MIB))
                / MIB,
        };
        self.cache_ceil_mb.map_or(mb, |cap| mb.min(cap)).saturating_mul(MIB)
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
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

fn parse_cache_mb(value: &str) -> Option<usize> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("auto") {
        None
    } else {
        value.parse::<usize>().ok()
    }
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

#[derive(Default)]
struct IoCounters {
    read_bytes: AtomicU64,
    read_nanos: AtomicU64,
    jobs: AtomicU64,
}

struct IoState {
    counters: IoCounters,
    effective_o_direct: AtomicBool,
}

impl Default for IoState {
    fn default() -> Self {
        Self {
            counters: IoCounters::default(),
            effective_o_direct: AtomicBool::new(false),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    source: String,
    expert: usize,
    row_start: usize,
    rows: usize,
}

struct CacheEntry {
    weight: Arc<dyn QuantMethod>,
    bytes: usize,
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<CacheKey, CacheEntry>,
    lru: VecDeque<CacheKey>,
    seen: HashSet<CacheKey>,
    resident_bytes: usize,
    lookups: u64,
    hits: u64,
    evictions: u64,
    rereads: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MoeStreamStatsSnapshot {
    pub read_bytes: u64,
    pub read_ms: f64,
    pub io_jobs: u64,
    pub lookups: u64,
    pub hits: u64,
    pub evictions: u64,
    pub rereads: u64,
    pub resident_bytes: usize,
    pub cache_budget_bytes: usize,
    pub effective_o_direct: bool,
}

struct ReadJob {
    shard: usize,
    offset: u64,
    len: usize,
    file_len: usize,
    reply: mpsc::Sender<io::Result<Vec<u8>>>,
}

pub(super) struct MoeStreamCache {
    budget_bytes: usize,
    config: MoeStreamConfig,
    paths: Arc<Vec<PathBuf>>,
    tx: SyncSender<ReadJob>,
    state: Mutex<CacheState>,
    io: Arc<IoState>,
    last_logged_lookups: AtomicU64,
}

static CACHE_REGISTRY: OnceLock<Mutex<HashMap<usize, Weak<MoeStreamCache>>>> = OnceLock::new();

impl MoeStreamCache {
    pub(super) fn for_archive(archive: Arc<GgufArchive>, config: MoeStreamConfig) -> Arc<Self> {
        let key = Arc::as_ptr(&archive) as usize;
        let registry = CACHE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));

        if let Some(existing) = registry
            .lock()
            .expect("MoE stream cache registry poisoned")
            .get(&key)
            .and_then(Weak::upgrade)
        {
            return existing;
        }

        let paths = archive
            .shards()
            .iter()
            .map(|shard| shard.path().to_path_buf())
            .collect::<Vec<_>>();

        let (tx, rx) =
            mpsc::sync_channel::<ReadJob>(config.io_threads.saturating_mul(4).max(4));
        let shared_rx = Arc::new(Mutex::new(rx));
        let io = Arc::new(IoState::default());
        io.effective_o_direct.store(config.o_direct, Ordering::Relaxed);

        let cache = Arc::new(Self {
            budget_bytes: config.cache_budget_bytes(),
            config,
            paths: Arc::new(paths),
            tx,
            state: Mutex::new(CacheState::default()),
            io: io.clone(),
            last_logged_lookups: AtomicU64::new(0),
        });

        for worker_id in 0..config.io_threads {
            let rx = shared_rx.clone();
            let paths = cache.paths.clone();
            let worker_io = io.clone();
            thread::Builder::new()
                .name(format!("mistralrs-moe-io-{worker_id}"))
                .spawn(move || io_worker(rx, &paths, &worker_io, config.o_direct))
                .expect("failed to spawn MoE streaming I/O worker");
        }

        registry
            .lock()
            .expect("MoE stream cache registry poisoned")
            .insert(key, Arc::downgrade(&cache));

        tracing::info!(
            cache_mb = cache.budget_bytes / MIB,
            io_threads = config.io_threads,
            overlap = config.overlap,
            o_direct = config.o_direct,
            "MoE expert streaming enabled"
        );

        cache
    }

    fn lookup(&self, key: &CacheKey) -> Option<Arc<dyn QuantMethod>> {
        let mut state = self.state.lock().expect("MoE stream cache poisoned");
        state.lookups = state.lookups.saturating_add(1);

        let Some(entry) = state.entries.get(key) else {
            if state.seen.contains(key) {
                state.rereads = state.rereads.saturating_add(1);
            }
            self.maybe_log(&state);
            return None;
        };

        let weight = entry.weight.clone();
        state.lru.retain(|candidate| candidate != key);
        state.lru.push_back(key.clone());
        state.hits = state.hits.saturating_add(1);
        self.maybe_log(&state);
        Some(weight)
    }

    fn insert(&self, key: CacheKey, weight: Arc<dyn QuantMethod>, bytes: usize) {
        if self.budget_bytes == 0 || bytes > self.budget_bytes {
            return;
        }

        let mut state = self.state.lock().expect("MoE stream cache poisoned");

        if let Some(old) = state.entries.remove(&key) {
            state.resident_bytes = state.resident_bytes.saturating_sub(old.bytes);
            state.lru.retain(|candidate| candidate != &key);
        }

        while state.resident_bytes.saturating_add(bytes) > self.budget_bytes {
            let Some(old_key) = state.lru.pop_front() else {
                break;
            };
            if let Some(old) = state.entries.remove(&old_key) {
                state.resident_bytes = state.resident_bytes.saturating_sub(old.bytes);
                state.evictions = state.evictions.saturating_add(1);
            }
        }

        if state.resident_bytes.saturating_add(bytes) <= self.budget_bytes {
            state.resident_bytes = state.resident_bytes.saturating_add(bytes);
            state.seen.insert(key.clone());
            state.entries.insert(key.clone(), CacheEntry { weight, bytes });
            state.lru.push_back(key);
        }
    }

    fn submit(
        &self,
        shard: usize,
        offset: u64,
        len: usize,
        file_len: usize,
    ) -> io::Result<Receiver<io::Result<Vec<u8>>>> {
        let (reply, rx) = mpsc::channel();
        self.tx
            .send(ReadJob {
                shard,
                offset,
                len,
                file_len,
                reply,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "MoE I/O worker stopped"))?;
        Ok(rx)
    }

    fn snapshot(&self) -> MoeStreamStatsSnapshot {
        let state = self.state.lock().expect("MoE stream cache poisoned");
        MoeStreamStatsSnapshot {
            read_bytes: self.io.counters.read_bytes.load(Ordering::Relaxed),
            read_ms: self.io.counters.read_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            io_jobs: self.io.counters.jobs.load(Ordering::Relaxed),
            lookups: state.lookups,
            hits: state.hits,
            evictions: state.evictions,
            rereads: state.rereads,
            resident_bytes: state.resident_bytes,
            cache_budget_bytes: self.budget_bytes,
            effective_o_direct: self.io.effective_o_direct.load(Ordering::Relaxed),
        }
    }

    fn maybe_log(&self, state: &CacheState) {
        if !self.config.stats || state.lookups == 0 || !state.lookups.is_multiple_of(256) {
            return;
        }
        let previous = self
            .last_logged_lookups
            .swap(state.lookups, Ordering::Relaxed);
        if previous == state.lookups {
            return;
        }

        tracing::info!(
            lookups = state.lookups,
            hits = state.hits,
            hit_rate = state.hits as f64 / state.lookups as f64,
            read_mb = self.io.counters.read_bytes.load(Ordering::Relaxed) as f64 / MIB as f64,
            read_ms = self.io.counters.read_nanos.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            io_jobs = self.io.counters.jobs.load(Ordering::Relaxed),
            evictions = state.evictions,
            rereads = state.rereads,
            resident_mb = state.resident_bytes as f64 / MIB as f64,
            cache_budget_mb = self.budget_bytes as f64 / MIB as f64,
            o_direct = self.io.effective_o_direct.load(Ordering::Relaxed),
            "MoE streaming telemetry"
        );
    }
}

impl Drop for MoeStreamCache {
    fn drop(&mut self) {
        let snapshot = self.snapshot();
        if snapshot.io_jobs == 0 && snapshot.lookups == 0 {
            return;
        }
        tracing::debug!(
            read_mb = snapshot.read_bytes as f64 / MIB as f64,
            read_ms = snapshot.read_ms,
            io_jobs = snapshot.io_jobs,
            lookups = snapshot.lookups,
            hits = snapshot.hits,
            evictions = snapshot.evictions,
            rereads = snapshot.rereads,
            resident_mb = snapshot.resident_bytes as f64 / MIB as f64,
            cache_budget_mb = snapshot.cache_budget_bytes as f64 / MIB as f64,
            o_direct = snapshot.effective_o_direct,
            "MoE streaming final telemetry"
        );
    }
}

fn io_worker(
    rx: Arc<Mutex<Receiver<ReadJob>>>,
    paths: &[PathBuf],
    io: &IoState,
    direct_requested: bool,
) {
    let mut files = Vec::with_capacity(paths.len());
    let mut direct_modes = Vec::with_capacity(paths.len());

    for path in paths {
        if direct_requested {
            match open_io_file(path, true) {
                Ok(file) => {
                    files.push(Some(file));
                    direct_modes.push(true);
                }
                Err(_) => {
                    // O_DIRECT is best-effort. Some filesystems (or mounts) reject it.
                    // Fall back to buffered reads for that shard instead of disabling the lane.
                    files.push(open_io_file(path, false).ok());
                    direct_modes.push(false);
                    io.effective_o_direct.store(false, Ordering::Relaxed);
                }
            }
        } else {
            files.push(open_io_file(path, false).ok());
            direct_modes.push(false);
        }
    }

    loop {
        let job = {
            let receiver = rx.lock().expect("MoE stream receive queue poisoned");
            match receiver.recv() {
                Ok(job) => job,
                Err(_) => return,
            }
        };

        let started = std::time::Instant::now();
        io.counters.jobs.fetch_add(1, Ordering::Relaxed);

        let result = if job.shard >= files.len() {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GGUF shard index out of range",
            ))
        } else if let Some(file) = files[job.shard].as_mut() {
            let direct = direct_modes[job.shard];
            let result = read_job_file(
                file,
                job.offset,
                job.len,
                job.file_len,
                direct,
            );
            if direct && result.is_err() {
                io.effective_o_direct.store(false, Ordering::Relaxed);
                match open_io_file(&paths[job.shard], false) {
                    Ok(mut fallback) => read_job_file(
                        &mut fallback,
                        job.offset,
                        job.len,
                        job.file_len,
                        false,
                    ),
                    Err(_) => result,
                }
            } else {
                result
            }
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "GGUF shard file is unavailable",
            ))
        };

        if let Ok(ref bytes) = result {
            io.counters
                .read_bytes
                .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        }
        io.counters.read_nanos.fetch_add(
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        let _ = job.reply.send(result);
    }
}

fn open_io_file(path: &Path, direct: bool) -> io::Result<File> {
    #[cfg(target_os = "linux")]
    if direct {
        return OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path);
    }

    OpenOptions::new().read(true).open(path)
}

fn read_job_file(
    file: &mut File,
    offset: u64,
    len: usize,
    file_len: usize,
    direct: bool,
) -> io::Result<Vec<u8>> {
    if !direct {
        let mut data = vec![0u8; len];
        read_exact_at(file, offset, &mut data)?;
        return Ok(data);
    }

    let aligned_start = offset / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT;
    let requested_end = offset
        .checked_add(len as u64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read range overflow"))?;
    let aligned_end = requested_end
        .div_ceil(DIRECT_ALIGNMENT)
        .saturating_mul(DIRECT_ALIGNMENT)
        .min(file_len as u64);

    if aligned_end < requested_end || aligned_end <= aligned_start {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "direct-I/O range is outside the GGUF shard",
        ));
    }

    let aligned_len = usize::try_from(aligned_end - aligned_start)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "direct-I/O range too large"))?;
    let mut aligned = AlignedBuffer::new(aligned_len, DIRECT_ALIGNMENT as usize)?;
    read_exact_at(file, aligned_start, aligned.as_mut_slice())?;

    let begin = usize::try_from(offset - aligned_start)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "direct-I/O offset overflow"))?;
    let end = begin
        .checked_add(len)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "direct-I/O slice overflow"))?;

    if end > aligned.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "direct-I/O read did not cover requested bytes",
        ));
    }

    Ok(aligned.as_slice()[begin..end].to_vec())
}

struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
}

unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(len: usize, align: usize) -> io::Result<Self> {
        let layout = std::alloc::Layout::from_size_align(len.max(1), align)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid aligned layout"))?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "aligned I/O allocation failed",
            ));
        }
        Ok(Self { ptr, len, layout })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let read = file.read_at(&mut buf[done..], offset)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short GGUF expert read",
            ));
        }
        done += read;
        offset += read as u64;
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut clone = file.try_clone()?;
    clone.seek(SeekFrom::Start(offset))?;
    clone.read_exact(buf)
}

#[derive(Clone, Copy, Debug)]
enum ProjectionFormat {
    Gguf(GgmlDType),
    Mxfp4,
}

enum PendingSource {
    Ready(Arc<dyn QuantMethod>),
    Io(Receiver<io::Result<Vec<u8>>>),
}

struct PendingExpert {
    key: CacheKey,
    source: PendingSource,
    format: ProjectionFormat,
    rows: usize,
    cols: usize,
    row_start: usize,
    bias: Option<Arc<Tensor>>,
    cache: Arc<MoeStreamCache>,
}

impl PendingExpert {
    fn wait(self, device: &Device) -> Result<Arc<dyn QuantMethod>> {
        let PendingExpert {
            key,
            source,
            format,
            rows,
            cols,
            row_start,
            bias,
            cache,
        } = self;

        let bytes = match source {
            PendingSource::Ready(weight) => return Ok(weight),
            PendingSource::Io(receiver) => receiver
                .recv()
                .map_err(|_| candle_core::Error::Msg("MoE expert I/O worker stopped".to_string()))?
                .map_err(|err| {
                    candle_core::Error::Msg(format!(
                        "MoE expert read failed for {} expert {}: {err}",
                        key.source, key.expert
                    ))
                })?,
        };

        let bias = bias
            .bias
            .as_ref()
            .map(|bias| {
                let bias = bias.narrow(0, key.expert, 1)?;
                let bias = bias.squeeze(0)?;
                if bias.rank() == 1 && row_start.saturating_add(rows) <= bias.dim(0)? {
                    bias.narrow(0, row_start, rows)
                } else {
                    Ok(bias)
                }
            })
            .transpose()?;

        let weight: Arc<dyn QuantMethod> = match self.format {
            ProjectionFormat::Gguf(dtype) => Arc::new(GgufMatMul::from_gguf_bytes(
                dtype,
                &bytes,
                vec![rows, cols],
                bias,
                device,
            )?),
            ProjectionFormat::Mxfp4 => Arc::new(MXFP4Layer::from_gguf_bytes(
                &bytes,
                rows,
                cols,
                bias,
                device,
            )?),
        };

        cache.insert(key, weight.clone(), bytes.len());
        Ok(weight)
    }
}

pub(super) struct StreamedProjection {
    archive: Arc<GgufArchive>,
    cache: Arc<MoeStreamCache>,
    source: String,
    shard: usize,
    tensor_offset: u64,
    file_len: usize,
    expert_count: usize,
    expert_bytes: usize,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    row_start: usize,
    format: ProjectionFormat,
    bias: Option<Arc<Tensor>>,
}

impl StreamedProjection {
    fn new(
        archive: Arc<GgufArchive>,
        cache: Arc<MoeStreamCache>,
        source: &str,
        row_start: usize,
        rows: usize,
        bias_name: Option<&str>,
    ) -> Result<Option<Self>> {
        let info = match archive.tensor_info(source) {
            Ok(info) => info,
            Err(_) => return Ok(None),
        };
        if info.shape().len() != 3 {
            return Ok(None);
        }

        let expert_count = info.shape()[0];
        let full_rows = info.shape()[1];
        let cols = info.shape()[2];
        if expert_count == 0
            || cols == 0
            || rows == 0
            || row_start
                .checked_add(rows)
                .is_none_or(|end| end > full_rows)
        {
            return Ok(None);
        }

        let format = if info.dtype().raw() == 39 {
            ProjectionFormat::Mxfp4
        } else {
            ProjectionFormat::Gguf(info.dtype().candle_dtype()?)
        };

        let data_len = info.byte_len().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "GGUF expert tensor {source} has no exact data range"
            ))
        })?;
        if !data_len.is_multiple_of(expert_count) {
            return Ok(None);
        }

        let expert_bytes = data_len / expert_count;
        if !expert_bytes.is_multiple_of(full_rows) {
            return Ok(None);
        }
        let row_bytes = expert_bytes / full_rows;

        let shard = info.shard_index();
        let shard_info = archive.shards().get(shard).ok_or_else(|| {
            candle_core::Error::Msg("GGUF expert shard index out of range".to_string())
        })?;
        let tensor_offset = info.data_range().ok_or_else(|| {
            candle_core::Error::Msg(format!("GGUF expert tensor {source} has no data range"))
        })?.start;

        let tensor_offset = u64::try_from(
            tensor_offset
                .checked_add(row_start.checked_mul(row_bytes).ok_or_else(|| {
                    candle_core::Error::Msg("GGUF expert row offset overflow".to_string())
                })?)
                .ok_or_else(|| {
                    candle_core::Error::Msg("GGUF expert byte offset overflow".to_string())
                })?,
        )
        .map_err(|_| candle_core::Error::Msg("GGUF expert offset exceeds u64".to_string()))?;

        let bias = match bias_name {
            Some(name) if archive.contains_tensor(name) => {
                let bias = archive
                    .load_qtensor(name, &Device::Cpu)?
                    .dequantize(&Device::Cpu)?;
                if bias.rank() != 2
                    || bias.dim(0)? != expert_count
                    || bias.dim(1)? < row_start.saturating_add(rows)
                {
                    return Ok(None);
                }
                Some(Arc::new(bias))
            }
            _ => None,
        };

        Ok(Some(Self {
            archive,
            cache,
            source: source.to_owned(),
            shard,
            tensor_offset,
            file_len: shard_info.file_len(),
            expert_count,
            expert_bytes,
            rows,
            cols,
            row_bytes,
            row_start,
            format,
            bias,
        }))
    }

    fn key(&self, expert: usize) -> CacheKey {
        CacheKey {
            source: self.source.clone(),
            expert,
            row_start: self.row_start,
            rows: self.rows,
        }
    }

    fn schedule(&self, expert: usize) -> Result<PendingExpert> {
        if expert >= self.expert_count {
            candle_core::bail!(
                "router selected expert {} for tensor {} with {} experts",
                expert,
                self.source,
                self.expert_count
            );
        }

        let key = self.key(expert);
        if let Some(weight) = self.cache.lookup(&key) {
            return Ok(PendingExpert {
                key,
                source: PendingSource::Ready(weight),
                format: self.format,
                rows: self.rows,
                cols: self.cols,
                row_start: self.row_start,
                bias: self.bias.clone(),
                cache: self.cache.clone(),
            });
        }

        let offset = self
            .tensor_offset
            .checked_add(expert.checked_mul(self.expert_bytes).ok_or_else(|| {
                candle_core::Error::Msg("MoE expert offset overflow".to_string())
            })?)
            .ok_or_else(|| candle_core::Error::Msg("MoE expert offset overflow".to_string()))?;

        let len = self
            .rows
            .checked_mul(self.row_bytes)
            .ok_or_else(|| candle_core::Error::Msg("MoE expert length overflow".to_string()))?;

        let receiver = self
            .cache
            .submit(self.shard, offset, len, self.file_len)?;

        Ok(PendingExpert {
            key,
            source: PendingSource::Io(receiver),
            format: self.format,
            rows: self.rows,
            cols: self.cols,
            row_start: self.row_start,
            bias: self.bias.clone(),
            cache: self.cache.clone(),
        })
    }

    fn plan(&self, ids: &[u32], top_k: usize) -> Result<ProjectionPlan> {
        let groups = group_routes(ids, top_k);
        let mut entries = Vec::with_capacity(groups.len());

        for (expert, routes, token_indices) in groups {
            entries.push(ProjectionPlanEntry {
                expert,
                routes,
                token_indices,
                pending: if self.cache.config.overlap {
                    Some(self.schedule(expert)?)
                } else {
                    None
                },
            });
        }

        Ok(ProjectionPlan { entries })
    }

    fn execute(
        &self,
        plan: ProjectionPlan,
        x: &Tensor,
        input_is_routed: bool,
        device: &Device,
    ) -> Result<Tensor> {
        let mut output = Tensor::zeros((x.dim(0)?, self.rows), x.dtype(), device)?;

        for entry in plan.entries {
            let pending = match entry.pending {
                Some(pending) => pending,
                None => self.schedule(entry.expert)?,
            };
            let weight = pending.wait(device)?;

            let input = if input_is_routed {
                &entry.routes
            } else {
                &entry.token_indices
            };

            let input_indices = Tensor::from_vec(
                input.iter().map(|v| *v as u32).collect::<Vec<_>>(),
                input.len(),
                device,
            )?;
            let output_indices = Tensor::from_vec(
                entry.routes.iter().map(|v| *v as u32).collect::<Vec<_>>(),
                entry.routes.len(),
                device,
            )?;

            let selected = x.index_select(&input_indices, 0)?;
            let projected = weight.forward(&selected)?;
            output = output.index_add(&output_indices, &projected, 0)?;
        }

        Ok(output)
    }
}

struct ProjectionPlanEntry {
    expert: usize,
    routes: Vec<usize>,
    token_indices: Vec<usize>,
    pending: Option<PendingExpert>,
}

struct ProjectionPlan {
    entries: Vec<ProjectionPlanEntry>,
}

fn group_routes(
    ids: &[u32],
    top_k: usize,
) -> Vec<(usize, Vec<usize>, Vec<usize>)> {
    let mut groups: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
    for (route, &expert) in ids.iter().enumerate() {
        let expert = expert as usize;
        if let Some((_, routes, tokens)) =
            groups.iter_mut().find(|(id, _, _)| *id == expert)
        {
            routes.push(route);
            tokens.push(route / top_k);
        } else {
            groups.push((expert, vec![route], vec![route / top_k]));
        }
    }
    groups
}

pub(super) struct StreamedExpertsWeights {
    gate: StreamedProjection,
    up: StreamedProjection,
    down: StreamedProjection,
}

impl StreamedExpertsWeights {
    pub(super) fn try_new(
        cfg: &super::config::MoEExpertsConfig,
        experts_vb: &mistralrs_quant::ShardedVarBuilder,
        layer_device: &Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Option<Self>> {
        let config = MoeStreamConfig::from_env();
        if !config.enabled
            || loading_isq
            || comm.world_size() != 1
            || !layer_device.is_cpu()
            || experts_vb.lora_registry().is_some()
        {
            return Ok(None);
        }

        let Some(archive) = experts_vb.raw_gguf() else {
            return Ok(None);
        };
        let Some(layer) =
            mistralrs_quant::layer_index_from_prefix(&experts_vb.prefix())
        else {
            return Ok(None);
        };

        let cache = MoeStreamCache::for_archive(archive.clone(), config);

        let split = |role: &str| format!("blk.{layer}.ffn_{role}_exps.weight");
        let split_bias = |role: &str| format!("blk.{layer}.ffn_{role}_exps.bias");

        let (gate, up) =
            if archive.contains_tensor(&split("gate"))
                && archive.contains_tensor(&split("up"))
            {
                (
                    StreamedProjection::new(
                        archive.clone(),
                        cache.clone(),
                        &split("gate"),
                        0,
                        cfg.moe_intermediate_size,
                        Some(&split_bias("gate")),
                    )?,
                    StreamedProjection::new(
                        archive.clone(),
                        cache.clone(),
                        &split("up"),
                        0,
                        cfg.moe_intermediate_size,
                        Some(&split_bias("up")),
                    )?,
                )
            } else {
                let fused = format!("blk.{layer}.ffn_gate_up_exps.weight");
                if !archive.contains_tensor(&fused) {
                    return Ok(None);
                }
                let bias = format!("blk.{layer}.ffn_gate_up_exps.bias");
                (
                    StreamedProjection::new(
                        archive.clone(),
                        cache.clone(),
                        &fused,
                        0,
                        cfg.moe_intermediate_size,
                        Some(&bias),
                    )?,
                    StreamedProjection::new(
                        archive.clone(),
                        cache.clone(),
                        &fused,
                        cfg.moe_intermediate_size,
                        cfg.moe_intermediate_size,
                        Some(&bias),
                    )?,
                )
            };

        let (Some(gate), Some(up)) = (gate, up) else {
            return Ok(None);
        };

        let down_name = split("down");
        let Some(down) = StreamedProjection::new(
            archive.clone(),
            cache,
            &down_name,
            0,
            cfg.hidden_size,
            Some(&split_bias("down")),
        )? else {
            return Ok(None);
        };

        if gate.expert_count != cfg.num_experts
            || up.expert_count != cfg.num_experts
            || down.expert_count != cfg.num_experts
            || gate.cols != cfg.hidden_size
            || up.cols != cfg.hidden_size
            || down.cols != cfg.moe_intermediate_size
            || gate.rows != cfg.moe_intermediate_size
            || up.rows != cfg.moe_intermediate_size
            || down.rows != cfg.hidden_size
        {
            tracing::warn!("GGUF MoE streaming shape mismatch; falling back to resident experts");
            return Ok(None);
        }

        Ok(Some(Self { gate, up, down }))
    }

    fn forward_impl(
        &self,
        forward: &super::forward::MoEForward,
        config: super::forward::MoEForwardConfig,
    ) -> Result<Tensor> {
        if forward.lora.is_some() {
            candle_core::bail!("dynamic LoRA is incompatible with streamed MoE experts");
        }

        let ids = forward.topk_ids.flatten_all()?.to_vec1::<u32>()?;
        let top_k = config.num_experts_per_tok;
        if top_k == 0 || ids.len() != forward.shape.num_tokens * top_k {
            candle_core::bail!("invalid streamed MoE routing shape");
        }

        let device = forward.xs_flat.device();
        let gate_plan = self.gate.plan(&ids, top_k)?;
        let up_plan = self.up.plan(&ids, top_k)?;
        let gate = self
            .gate
            .execute(gate_plan, forward.xs_flat, false, device)?;
        let up = self
            .up
            .execute(up_plan, forward.xs_flat, false, device)?;
        let down_input = up.mul(&gate.apply(&config.act)?)?;

        let down_plan = self.down.plan(&ids, top_k)?;
        let down = self.down.execute(
            down_plan,
            &down_input.reshape((forward.shape.num_tokens * top_k, self.down.cols))?,
            true,
            device,
        )?;

        down.reshape((forward.shape.num_tokens, top_k, self.down.rows))?
            .to_dtype(DType::F32)?
            .broadcast_mul(&forward.topk_weights.unsqueeze(candle_core::D::Minus1)?)?
            .sum(candle_core::D::Minus2)?
            .to_dtype(forward.original_dtype)
    }

    #[allow(dead_code)]
    pub(super) fn stats(&self) -> MoeStreamStatsSnapshot {
        self.gate.cache.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cache_spec() {
        assert_eq!(parse_cache_mb("auto"), None);
        assert_eq!(parse_cache_mb("AUTO"), None);
        assert_eq!(parse_cache_mb("0"), Some(0));
        assert_eq!(parse_cache_mb("2048"), Some(2048));
        assert_eq!(parse_cache_mb("bogus"), None);
    }

    #[test]
    fn groups_routed_positions_without_losing_route_order() {
        let groups = group_routes(&[2, 1, 2, 0, 1, 0], 2);
        assert_eq!(groups[0].0, 2);
        assert_eq!(groups[0].1, vec![0, 2]);
        assert_eq!(groups[0].2, vec![0, 1]);
        assert_eq!(groups[1].0, 1);
        assert_eq!(groups[1].1, vec![1, 4]);
        assert_eq!(groups[1].2, vec![0, 2]);
    }

    #[test]
    fn auto_cache_budget_respects_floor_and_cap() {
        let cfg = MoeStreamConfig {
            enabled: true,
            cache_mb: Some(8192),
            cache_floor_mb: 1024,
            cache_ceil_mb: Some(2048),
            io_threads: 4,
            overlap: true,
            o_direct: false,
            stats: false,
        };
        assert_eq!(cfg.cache_budget_bytes(), 2048 * MIB);
    }
}
