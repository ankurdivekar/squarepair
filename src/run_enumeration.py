"""Parallel driver for full square-sum pair enumeration.

Splits the deterministic search tree into a frontier of partial matchings
("prefixes"), farms them to a process pool, and has each worker stream its
solutions straight to a shard CSV -- solution rows never cross the process
boundary. Shards are then concatenated into the final ``Pair No,Pair 1,...`` CSV.

Run under PyPy for the heavy sizes::

    uv run --python pypy3.11 -m src.run_enumeration --n 100 \\
        --out data/complete_sets_n100.csv --workers 16 --shard-dir /scratch/n100

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from src.csv_writer import merge_shards
from src.pair_solver import (
    Pair,
    Solution,
    _ensure_recursion_limit,
    _popcount,
    build_adjacency,
    solve_prefix,
)

# Worker globals, populated once per process by the pool initializer.
_ADJ: list[int] | None = None
_FULL: int = 0
_SPLIT: bool = False


def _init_worker(adj: list[int], full_mask: int, split: bool) -> None:
    global _ADJ, _FULL, _SPLIT
    _ADJ, _FULL, _SPLIT = adj, full_mask, split
    _ensure_recursion_limit(full_mask.bit_length())


def _first_free(free: int) -> int:
    return (free & -free).bit_length() - 1


def generate_prefixes(
    adj: list[int], full_mask: int, target_tasks: int, max_depth: int = 14
) -> list[tuple[Pair, ...]]:
    """Breadth-first expansion of the same deterministic tree the solver walks
    (forced-move propagation, then MRV branching), stopped once the frontier holds
    at least ``target_tasks`` nodes. Every complete matching extends exactly one
    returned prefix, so the prefixes partition the solution space.
    """
    # frontier entries: (free_mask, prefix_pairs)
    frontier: list[tuple[int, tuple[Pair, ...]]] = [(full_mask, ())]
    done: list[tuple[Pair, ...]] = []  # leaf prefixes (free == 0)

    while len(frontier) + len(done) < target_tasks:
        # expand the open node with the largest free set (biggest expected subtree)
        open_idx = -1
        best_pc = -1
        for i, (fm, _) in enumerate(frontier):
            pc = _popcount(fm)
            if pc > best_pc:
                best_pc = pc
                open_idx = i
        if open_idx < 0:
            break
        free, prefix = frontier.pop(open_idx)

        free, prefix, status, pivot = _expand_one(adj, free, prefix)
        if status == "dead":
            continue
        if status == "leaf":
            done.append(prefix)
            continue
        if len(prefix) >= max_depth:
            frontier.append((free, prefix))  # stop splitting this path
            # if every remaining open node is at max depth we'd loop forever
            if all(len(p) >= max_depth for _, p in frontier):
                break
            continue

        bit_v = 1 << pivot
        m = adj[pivot] & free
        while m:
            b = m & -m
            m ^= b
            p = b.bit_length() - 1
            pair: Pair = (pivot, p) if pivot < p else (p, pivot)
            frontier.append((free ^ bit_v ^ b, prefix + (pair,)))

    return [p for _, p in frontier] + done


def _expand_one(
    adj: list[int], free: int, prefix: tuple[Pair, ...]
) -> tuple[int, tuple[Pair, ...], str, int]:
    """Run forced-move propagation on ``free`` (mirrors ``pair_solver._rec``).

    Returns ``(free, prefix, status, pivot)`` where status is ``"dead"``,
    ``"leaf"`` (free == 0) or ``"branch"`` (pivot is the MRV vertex to split on).
    """
    added: list[Pair] = []
    while True:
        m = free
        deg1_v = -1
        deg1_p = 0
        min_deg = 1 << 30
        min_v = -1
        dead = False
        while m:
            b = m & -m
            m ^= b
            v = b.bit_length() - 1
            avail = adj[v] & free
            d = _popcount(avail)
            if d == 0:
                dead = True
                break
            if d == 1:
                deg1_v = v
                deg1_p = avail.bit_length() - 1
                break
            if d < min_deg:
                min_deg = d
                min_v = v
        if dead:
            return free, prefix, "dead", -1
        if deg1_v < 0:
            break
        free ^= (1 << deg1_v) | (1 << deg1_p)
        added.append((deg1_v, deg1_p) if deg1_v < deg1_p else (deg1_p, deg1_v))

    prefix = prefix + tuple(added)
    if free == 0:
        return free, prefix, "leaf", -1
    if _popcount(free) & 1:
        return free, prefix, "dead", -1
    return free, prefix, "branch", min_v


def _ordered(pairs: Solution) -> list[str]:
    return [f"{a}-{b}" if a < b else f"{b}-{a}" for a, b in pairs]


def _run_task(task_id: int, prefix: tuple[Pair, ...], shard_dir: str) -> tuple[int, str, int]:
    assert _ADJ is not None
    path = os.path.join(shard_dir, f"shard_{task_id:05d}.csv")
    buf: list[str] = []
    stats = [0]  # solution count

    def emit(sol: Solution) -> None:
        stats[0] += 1
        buf.append(",".join(_ordered(sol)))
        buf.append("\n")
        if len(buf) >= 8192:
            fh.write("".join(buf))
            buf.clear()

    with open(path, "w", newline="") as fh:
        solve_prefix(_ADJ, _FULL, prefix, emit, split=_SPLIT)
        if buf:
            fh.write("".join(buf))
    return task_id, path, stats[0]


def enumerate_to_csv(
    n_numbers: int,
    output_csv: str,
    *,
    workers: int | None = None,
    shard_dir: str | None = None,
    target_tasks: int | None = None,
    resume: bool = True,
    split: bool = False,
    count_only: bool = False,
    keep_shards: bool = False,
) -> int:
    """Enumerate every square-sum pair partition of ``1..n_numbers`` into ``output_csv``.

    Returns the total solution count.
    """
    adj, full = build_adjacency(n_numbers)
    n_pairs = n_numbers // 2
    workers = workers or (os.cpu_count() or 1)

    if count_only:
        from src.pair_solver import count_solutions

        start = time.time()
        total = count_solutions(n_numbers, split=split)
        print(f"n={n_numbers}: {total:,} solutions in {time.time() - start:.1f}s (count only)")
        return total

    target_tasks = target_tasks or workers * 16
    owns_shard_dir = shard_dir is None
    shard_dir = shard_dir or f"{output_csv}.shards"
    os.makedirs(shard_dir, exist_ok=True)

    prefixes = generate_prefixes(adj, full, target_tasks)
    print(f"n={n_numbers}: {len(prefixes)} prefixes over {workers} workers")

    manifest_path = os.path.join(shard_dir, "manifest.json")
    manifest: dict[str, dict] = {}
    if resume and os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    ctx = get_context("fork" if sys.platform != "win32" else "spawn")
    pending = [
        (i, p) for i, p in enumerate(prefixes) if not manifest.get(str(i), {}).get("done")
    ]
    if len(pending) < len(prefixes):
        print(f"  resuming: {len(prefixes) - len(pending)} shards already done")

    start = time.time()
    running_total = sum(m["count"] for m in manifest.values() if m.get("done"))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(adj, full, split),
        mp_context=ctx,
    ) as pool:
        futures = {
            pool.submit(_run_task, i, p, shard_dir): i for i, p in pending
        }
        completed = len(prefixes) - len(pending)
        for fut in as_completed(futures):
            task_id, path, count = fut.result()
            running_total += count
            completed += 1
            manifest[str(task_id)] = {"path": path, "count": count, "done": True}
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh)
            print(
                f"  [{completed}/{len(prefixes)}] shard {task_id}: {count:,} "
                f"(total {running_total:,}, {time.time() - start:.0f}s)"
            )

    shard_paths = [manifest[str(i)]["path"] for i in range(len(prefixes))]
    merge_shards(shard_paths, output_csv, n_pairs)
    print(f"n={n_numbers}: {running_total:,} solutions -> {output_csv} in {time.time() - start:.0f}s")

    if owns_shard_dir and not keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)
    return running_total


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="n_numbers (even): partition 1..n")
    ap.add_argument("--out", help="output CSV path (default data/complete_sets_n{N}.csv)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--shard-dir", default=None, help="scratch dir for shard files (kept if given)")
    ap.add_argument("--target-tasks", type=int, default=None, help="prefix count (default workers*16)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--split", action="store_true", help="enable experimental component split")
    ap.add_argument("--count-only", action="store_true", help="just count, no CSV")
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or f"data/complete_sets_n{args.n}.csv"
    enumerate_to_csv(
        args.n,
        out,
        workers=args.workers,
        shard_dir=args.shard_dir,
        target_tasks=args.target_tasks,
        resume=not args.no_resume,
        split=args.split,
        count_only=args.count_only,
        keep_shards=args.keep_shards or args.shard_dir is not None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
