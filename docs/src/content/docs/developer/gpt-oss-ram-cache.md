---
title: GPT-OSS MXFP4 RAM expert cache
description: Keep routed GPT-OSS MXFP4 expert bytes in a bounded RAM LRU instead of relying only on mmap page-cache residency.
sidebar:
  order: 7
---

# GPT-OSS MXFP4 RAM expert cache

The default GPT-OSS MXFP4 streaming path can use zero-copy mappings into the GGUF file. That is memory-efficient, but the cache can retain mmap descriptors while Linux repeatedly reclaims and reloads the underlying file pages. Its cache's `used_mib` can therefore stay at zero even while disk activity is high.

For an Edge0-inspired *owned RAM cache*, set `MISTRALRS_MOE_RAM_CACHE=1`. This switches GPT-OSS MXFP4 streaming to read routed expert byte ranges into owned buffers, retain them in a byte-bounded shared LRU, and drop least-recently-used buffers when the configured budget is exceeded. On Linux it requests `O_DIRECT` so the same expert data does not also compete for space in the kernel page cache; if direct I/O cannot be used, the reader falls back to buffered I/O. Entries still leased by an active computation are not selected for eviction.

The existing routed prefetch path is preserved: the exact experts selected by GPT-OSS's own router are queued for reading, and `--moe-overlap` lets reads overlap compute. This reuses Edge0's bounded resident-cache and background-prefetch design without pretending that Edge0's model-specific trained prerouter heads can be applied to GPT-OSS. A trained next-token routing predictor would need weights trained for the actual GPT-OSS architecture.

## Run on a low-memory Linux system

Start with a cache budget of 768 MiB on an 8 GiB machine:

```bash
MISTRALRS_MOE_RAM_CACHE=1 RUST_LOG=info /usr/bin/time -v ./mistralrs run \
  --cpu \
  --moe-stream \
  --cache-mb 768 \
  --cache-floor-mb 1024 \
  --cache-ceil-mb 768 \
  --io-threads 2 \
  --moe-overlap \
  --moe-stats \
  --max-model-len 2048 \
  --max-tokens 96 \
  --thinking false \
  -f "$HOME/.lmstudio/models/openai/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf" \
  -i "Explain how an operating system schedules processes in about 60 words." \
  > gpt-oss.stdout 2> gpt-oss.stderr
```

Check streaming telemetry in `gpt-oss.stderr`:

- `zero_copy=false` confirms the owned-buffer path is active.
- `o_direct=true` means direct I/O was requested; unsupported file systems or alignment errors may cause a buffered fallback.
- `used_mib` should rise above zero as owned expert buffers enter the LRU.
- `hits`, `misses`, and `evictions` show whether the cache budget is large enough for the working set.
- `Maximum resident set size` from `/usr/bin/time -v` reports peak process RSS.

A cache hit serves bytes from RAM; an evicted expert must be read from storage again. The budget applies to cache-retained bytes, not the model's full weight size or other process memory. On an 8 GiB system, increase the budget only while monitoring available RAM and swap pressure.

The design is inspired by [Edge0](https://github.com/Edge0-AI/Edge0), in particular its [shared expert LRU](https://github.com/Edge0-AI/Edge0/blob/main/python/src/edge0/streaming/cache.py) and [lease-aware RAM hot stack](https://github.com/Edge0-AI/Edge0/blob/main/macos/crates/engine-native/src/hotstack.cpp); this implementation remains native to mistral.rs's GPT-OSS MXFP4 reader.
