#!/usr/bin/env python3
"""Synthetic cold-page I/O benchmark for the MoE expert-cache policy.

This is an SSD-pressure proxy, not a GPT-OSS end-to-end benchmark. It creates a
small temporary file, simulates a skewed expert-routing stream, and compares
repeated file reads with a bounded in-RAM LRU. POSIX_FADV_DONTNEED is advisory;
the report includes logical file reads and Linux process storage-read bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


def process_storage_read_bytes() -> int | None:
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            if line.startswith("read_bytes:"):
                return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


def advise_drop(fd: int, offset: int, length: int) -> bool:
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    fn = getattr(os, "posix_fadvise", None)
    if advice is None or fn is None or length <= 0:
        return False
    try:
        fn(fd, offset, length, advice)
        return True
    except OSError:
        return False


def make_requests(count: int, experts: int, hot_experts: int,
                  hot_probability: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    result: list[int] = []
    for _ in range(count):
        if rng.random() < hot_probability:
            result.append(rng.randrange(hot_experts))
        else:
            result.append(rng.randrange(hot_experts, experts))
    return result


def run_case(path: Path, requests: list[int], expert_bytes: int,
             cache_capacity: int, cache_enabled: bool) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY)
    cache: OrderedDict[int, bytes] = OrderedDict()
    hits = misses = file_read_bytes = read_calls = fadvise_calls = 0
    advise_drop(fd, 0, path.stat().st_size)
    before = process_storage_read_bytes()
    started = time.perf_counter()
    try:
        for expert in requests:
            data = cache.get(expert) if cache_enabled else None
            if data is not None:
                hits += 1
                cache.move_to_end(expert)
                continue

            misses += 1
            offset = expert * expert_bytes
            data = os.pread(fd, expert_bytes, offset)
            if len(data) != expert_bytes:
                raise RuntimeError(f"short read for expert {expert}: {len(data)} bytes")
            read_calls += 1
            file_read_bytes += len(data)

            # Data has been copied into an owned Python bytes object. Ask the OS
            # to discard these clean file pages to approximate a cold-page read.
            if advise_drop(fd, offset, expert_bytes):
                fadvise_calls += 1

            if cache_enabled and cache_capacity > 0:
                cache[expert] = data
                cache.move_to_end(expert)
                while len(cache) > cache_capacity:
                    cache.popitem(last=False)
    finally:
        os.close(fd)
    elapsed = time.perf_counter() - started
    after = process_storage_read_bytes()
    physical = None if before is None or after is None else max(0, after - before)
    return {
        "mode": "bounded_lru" if cache_enabled else "uncached",
        "requests": len(requests),
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": round(100.0 * hits / max(1, len(requests)), 2),
        "file_read_calls": read_calls,
        "file_read_bytes_requested": file_read_bytes,
        "process_storage_read_bytes_delta": physical,
        "elapsed_seconds": round(elapsed, 4),
        "requests_per_second": round(len(requests) / max(elapsed, 1e-9), 1),
        "fadvise_dontneed_calls": fadvise_calls,
        "fadvise_supported": hasattr(os, "posix_fadvise"),
        "cache_capacity_experts": cache_capacity if cache_enabled else 0,
        "cache_bytes": cache_capacity * expert_bytes if cache_enabled else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--expert-kib", type=int, default=64)
    parser.add_argument("--requests", type=int, default=2048)
    parser.add_argument("--cache-experts", type=int, default=32)
    parser.add_argument("--hot-experts", type=int, default=16)
    parser.add_argument("--hot-probability", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.experts < 2 or args.expert_kib < 4 or args.requests < 1:
        parser.error("experts >= 2, expert-kib >= 4 and requests >= 1 are required")
    if not 1 <= args.hot_experts < args.experts:
        parser.error("hot-experts must be in [1, experts)")
    if not 0.0 <= args.hot_probability <= 1.0:
        parser.error("hot-probability must be in [0, 1]")
    if args.cache_experts < 1:
        parser.error("cache-experts must be >= 1")

    expert_bytes = args.expert_kib * 1024
    temp_root = os.environ.get("RUNNER_TEMP")
    kwargs = {"dir": temp_root} if temp_root and Path(temp_root).is_dir() else {}
    requests = make_requests(args.requests, args.experts, args.hot_experts,
                             args.hot_probability, args.seed)

    with tempfile.TemporaryDirectory(prefix="moe-ssd-bench-", **kwargs) as temp:
        path = Path(temp) / "synthetic-experts.bin"
        with path.open("wb") as out:
            remaining = args.experts * expert_bytes
            while remaining:
                chunk = os.urandom(min(1 << 20, remaining))
                out.write(chunk)
                remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())

        uncached = run_case(path, requests, expert_bytes, 0, False)
        cached = run_case(path, requests, expert_bytes, args.cache_experts, True)

    logical_saved = uncached["file_read_bytes_requested"] - cached["file_read_bytes_requested"]
    logical_reduction_pct = 100.0 * logical_saved / max(1, uncached["file_read_bytes_requested"])
    uncached_physical = uncached["process_storage_read_bytes_delta"]
    cached_physical = cached["process_storage_read_bytes_delta"]
    physical_reduction_pct = None
    if uncached_physical is not None and uncached_physical > 0 and cached_physical is not None:
        physical_reduction_pct = round(
            100.0 * (uncached_physical - cached_physical) / uncached_physical, 2
        )

    result = {
        "benchmark": "synthetic_moe_expert_io_pressure",
        "note": "Not a GPT-OSS end-to-end benchmark. process_storage_read_bytes_delta is process-wide and POSIX_FADV_DONTNEED is advisory.",
        "workload": {
            "experts": args.experts,
            "expert_kib": args.expert_kib,
            "requests": args.requests,
            "hot_experts": args.hot_experts,
            "hot_probability": args.hot_probability,
            "seed": args.seed,
        },
        "uncached": uncached,
        "bounded_lru": cached,
        "logical_file_read_reduction_pct": round(logical_reduction_pct, 2),
        "process_storage_read_reduction_pct": physical_reduction_pct,
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    if cached["file_read_bytes_requested"] >= uncached["file_read_bytes_requested"]:
        raise RuntimeError("bounded LRU did not reduce requested file bytes")
    if cached["cache_hits"] == 0:
        raise RuntimeError("synthetic workload produced no cache hits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
