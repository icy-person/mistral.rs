#!/usr/bin/env python3
"""Synthetic MoE expert I/O benchmark.

This is an SSD-I/O proxy, not an end-to-end model benchmark. It compares an
uncached reader with byte-bounded LRU caches across uniform and skewed routing,
with OS page-cache retention or best-effort POSIX_FADV_DONTNEED after each miss.
Linux read_bytes is process-wide; fadvise is advisory and may be ignored.
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


def make_requests(
    count: int,
    experts: int,
    hot_experts: int,
    hot_probability: float,
    seed: int,
    profile: str = "custom",
) -> list[int]:
    rng = random.Random(seed)
    result: list[int] = []
    for _ in range(count):
        if profile == "uniform":
            result.append(rng.randrange(experts))
        elif rng.random() < hot_probability:
            result.append(rng.randrange(hot_experts))
        else:
            result.append(rng.randrange(hot_experts, experts))
    return result


def run_case(
    path: Path,
    requests: list[int],
    expert_bytes: int,
    cache_capacity: int,
    drop_pages_after_miss: bool,
) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY)
    cache: OrderedDict[int, bytes] = OrderedDict()
    hits = misses = file_read_bytes = read_calls = 0

    # A fresh best-effort cold start for each scenario. Whether the kernel
    # actually drops the pages is deliberately measured, not assumed.
    initial_advice = advise_drop(fd, 0, path.stat().st_size)
    after_read_advice_calls = 0
    before = process_storage_read_bytes()
    started = time.perf_counter()
    try:
        for expert in requests:
            data = cache.get(expert) if cache_capacity > 0 else None
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

            if drop_pages_after_miss and advise_drop(fd, offset, expert_bytes):
                after_read_advice_calls += 1

            if cache_capacity > 0:
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
        "cache_capacity_experts": cache_capacity,
        "cache_capacity_mib": round(cache_capacity * expert_bytes / (1024 * 1024), 2),
        "page_policy": "drop_after_miss" if drop_pages_after_miss else "retain_after_start",
        "requests": len(requests),
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": round(100.0 * hits / max(1, len(requests)), 2),
        "file_read_calls": read_calls,
        "file_read_mib_requested": round(file_read_bytes / (1024 * 1024), 2),
        "process_storage_read_mib_delta": (
            None if physical is None else round(physical / (1024 * 1024), 2)
        ),
        "elapsed_seconds": round(elapsed, 4),
        "requests_per_second": round(len(requests) / max(elapsed, 1e-9), 1),
        "initial_dontneed_succeeded": initial_advice,
        "dontneed_after_miss_calls": after_read_advice_calls,
        "fadvise_supported": hasattr(os, "posix_fadvise"),
    }


def reduction_pct(baseline: float | int | None, value: float | int | None) -> float | None:
    if baseline is None or value is None or baseline <= 0:
        return None
    return round(100.0 * (baseline - value) / baseline, 2)


def print_table(results: list[dict[str, Any]]) -> None:
    headers = ("PROFILE", "PAGE POLICY", "CACHE", "HIT%", "READ MiB", "STORAGE MiB", "SEC", "REQ/S", "READ↓%")
    print("\n" + " | ".join(headers))
    print("-" * 126)
    for r in results:
        storage = r["process_storage_read_mib_delta"]
        storage_text = "n/a" if storage is None else f'{storage:.2f}'
        reduction = r.get("logical_read_reduction_pct")
        reduction_text = "—" if reduction is None else f"{reduction:.2f}"
        print(
            f'{r["profile"]:<8} | {r["page_policy"]:<17} | {r["cache_capacity_experts"]:>5} | '
            f'{r["hit_rate_pct"]:>5.1f} | {r["file_read_mib_requested"]:>8.2f} | '
            f'{storage_text:>11} | {r["elapsed_seconds"]:>6.3f} | '
            f'{r["requests_per_second"]:>7.1f} | {reduction_text:>6}'
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--expert-kib", type=int, default=64)
    parser.add_argument("--requests", type=int, default=2048)
    parser.add_argument("--cache-experts", type=int, default=32)
    parser.add_argument("--hot-experts", type=int, default=16)
    parser.add_argument("--hot-probability", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matrix", action="store_true", help="run routing/cache/page-policy comparison matrix")
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
    cache_capacities = [0, args.cache_experts]
    if args.matrix:
        # Chosen to compare tiny, moderate, large, and full-working-set caches.
        cache_capacities = sorted(set([0, 8, 16, 32, 64, args.experts]))
        profiles = [
            ("uniform", None),
            ("skew50", 0.50),
            ("skew85", 0.85),
            ("skew95", 0.95),
        ]
        hot_experts = min(args.hot_experts, args.experts - 1)
        policies = [False, True]
    else:
        profiles = [("custom", args.hot_probability)]
        hot_experts = args.hot_experts
        policies = [False, True]

    temp_root = os.environ.get("RUNNER_TEMP")
    kwargs = {"dir": temp_root} if temp_root and Path(temp_root).is_dir() else {}
    matrix_results: list[dict[str, Any]] = []

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

        for profile_index, (profile, probability) in enumerate(profiles):
            requests = make_requests(
                args.requests,
                args.experts,
                hot_experts,
                args.hot_probability if probability is None else probability,
                args.seed + profile_index,
                profile="uniform" if profile == "uniform" else "custom",
            )
            for drop_after_miss in policies:
                case_results: list[dict[str, Any]] = []
                for capacity in cache_capacities:
                    result = run_case(path, requests, expert_bytes, capacity, drop_after_miss)
                    result["profile"] = profile
                    case_results.append(result)

                baseline = next(r for r in case_results if r["cache_capacity_experts"] == 0)
                for result in case_results:
                    result["logical_read_reduction_pct"] = reduction_pct(
                        baseline["file_read_mib_requested"], result["file_read_mib_requested"]
                    )
                    result["storage_read_reduction_pct"] = reduction_pct(
                        baseline["process_storage_read_mib_delta"],
                        result["process_storage_read_mib_delta"],
                    )
                # LRU stack property: increasing capacity should not reduce hit rate.
                hits_by_capacity = [
                    (r["cache_capacity_experts"], r["cache_hits"]) for r in case_results
                ]
                if any(hits_by_capacity[i][1] > hits_by_capacity[i + 1][1]
                       for i in range(len(hits_by_capacity) - 1)):
                    raise RuntimeError(f"hit rate regressed as cache capacity increased for {profile}")
                if baseline["cache_hits"] != 0:
                    raise RuntimeError("uncached baseline unexpectedly reported cache hits")
                matrix_results.extend(case_results)

    payload = {
        "benchmark": "synthetic_moe_expert_io_comparison",
        "note": (
            "Proxy benchmark only; not a GPT-OSS end-to-end test. Storage read bytes are "
            "process-wide. POSIX_FADV_DONTNEED is advisory and the runner's virtualized "
            "storage/cache behavior may differ from a user's SSD."
        ),
        "workload": {
            "experts": args.experts,
            "expert_kib": args.expert_kib,
            "file_mib": round(args.experts * expert_bytes / (1024 * 1024), 2),
            "requests_per_profile": args.requests,
            "hot_experts": hot_experts,
            "cache_capacities_experts": cache_capacities,
            "profiles": [p[0] for p in profiles],
            "page_policies": ["retain_after_start", "drop_after_miss"],
            "seed": args.seed,
        },
        "results": matrix_results,
    }

    print_table(matrix_results)
    output = json.dumps(payload, indent=2, sort_keys=True)
    print("\nFull comparison JSON:")
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
