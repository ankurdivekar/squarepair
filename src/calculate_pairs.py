"""Generate complete sets of square-sum pairs for the numbers ``1..n_numbers``.

A "complete set" partitions ``1..n_numbers`` into pairs whose sums are all perfect
squares -- i.e. a perfect matching of the square-sum graph. The count as a function
of ``n_numbers // 2`` is OEIS A252897.

The search lives in :mod:`src.pair_solver` (bitmask backtracking with forced-move
propagation and MRV branching); the parallel, streaming-to-CSV driver lives in
:mod:`src.run_enumeration`. This module keeps the small, notebook-facing API.
"""

from __future__ import annotations

import time

from src.pair_solver import Pair, build_adjacency, count_solutions, iter_solutions

Solution = list[Pair]


def get_square_combinations(n_numbers: int) -> list[Pair]:
    """All pairs ``(i, j)`` with ``i < j <= n_numbers`` and ``i + j`` a perfect square."""
    squares: list[int] = []
    s = 2
    while s * s <= 2 * n_numbers - 1:
        squares.append(s * s)
        s += 1

    pairs: list[Pair] = []
    for i in range(1, n_numbers + 1):
        for sq in squares:
            j = sq - i
            if i < j <= n_numbers:
                pairs.append((i, j))
    return pairs


def generate_complete_sets(n_numbers: int, *, max_solutions: int | None = None) -> list[Solution]:
    """Return every complete set of square-sum pairs for ``1..n_numbers``.

    Materialises all solutions in memory -- fine for the notebook's default sizes
    (n <= ~60). For larger ``n_numbers`` use
    :func:`generate_complete_sets_to_csv`, which streams to disk in parallel.

    Args:
        n_numbers: Upper bound of the range (must be a positive even integer).
        max_solutions: If set, stop after this many solutions.
    """
    start = time.time()
    solutions: list[Solution] = []
    for sol in iter_solutions(n_numbers):
        solutions.append(sol)
        if max_solutions is not None and len(solutions) >= max_solutions:
            break

    print(
        f"Found {len(solutions)} complete sets for n={n_numbers} "
        f"in {time.time() - start:.2f}s"
    )
    return solutions


def generate_complete_sets_to_csv(
    n_numbers: int,
    output_csv: str,
    *,
    workers: int | None = None,
    shard_dir: str | None = None,
    target_tasks: int | None = None,
    resume: bool = True,
    count_only: bool = False,
) -> int:
    """Enumerate every complete set for ``1..n_numbers`` straight to ``output_csv``.

    Parallel and constant-memory: workers stream their solutions to shard files
    which are then concatenated. Returns the total solution count.
    """
    from src.run_enumeration import enumerate_to_csv

    return enumerate_to_csv(
        n_numbers,
        output_csv,
        workers=workers,
        shard_dir=shard_dir,
        target_tasks=target_tasks,
        resume=resume,
        count_only=count_only,
    )


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    out = sys.argv[2] if len(sys.argv) > 2 else f"data/complete_sets_n{n}.csv"
    total = generate_complete_sets_to_csv(n, out)
    print("\n" + 50 * "=")
    print(f"OEIS A252897 a({n // 2}) = {total}" if total else "No valid complete sets found.")
    print(50 * "=")
    # cross-check for small n
    if n <= 44:
        assert total == count_solutions(n), "parallel/serial mismatch"
