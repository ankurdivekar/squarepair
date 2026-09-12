"""Bitmask backtracking enumerator for square-sum pair partitions (OEIS A252897).

Given ``n_numbers`` (even), enumerate every way to partition ``1..n_numbers`` into
pairs whose sums are perfect squares -- equivalently, every perfect matching of the
graph with an edge ``i--j`` iff ``i + j`` is a perfect square.

The search is a recursive backtracker over an integer-bitmask state with the
prunings the original implementation lacked:

* **forced-move propagation** -- a still-free vertex with exactly one available
  partner is matched immediately, to a fixed point;
* **dead-end pruning** -- a free vertex with zero available partners, or an odd
  free set, kills the branch;
* **MRV branching** -- the pivot is the free vertex with the *fewest* available
  partners, not the smallest index.

There is also an opt-in ``split=True`` mode that decomposes a fragmented free set
into connected components and stitches their matchings with a Cartesian product.
It is off by default: on the square-sum graph up to n=60 it measured slower than
plain recursion (the components rarely fragment early), but it is kept for
experimentation at larger n.

Stdlib only, and written to run unchanged under PyPy.
"""

from __future__ import annotations

import itertools
import queue
import sys
import threading
from typing import Callable, Iterator

# A solution is a list of (a, b) pairs with a < b.
Pair = tuple[int, int]
Solution = list[Pair]
Emit = Callable[[Solution], None]

# Only hunt for component splits once the free set is small enough that the
# O(V + E) component scan is cheap relative to the subtree it may prune.
_SPLIT_THRESHOLD = 60

# If a single component would materialise more than this many solutions, abandon
# the split for this node and fall back to plain MRV recursion (bounds memory).
_SPLIT_MATERIALISE_CAP = 200_000

try:  # int.bit_count: CPython 3.10+, PyPy 7.3.12+
    _popcount = int.bit_count
    _popcount(0)
except (AttributeError, TypeError):  # pragma: no cover - old interpreters

    def _popcount(x: int) -> int:
        return bin(x).count("1")


def build_adjacency(n_numbers: int) -> tuple[list[int], int]:
    """Return ``(adj, full_mask)`` for the square-sum graph on ``1..n_numbers``.

    ``adj[v]`` is an int bitmask: bit ``p`` is set iff ``1 <= p <= n_numbers``,
    ``p != v`` and ``v + p`` is a perfect square. ``full_mask`` has bit ``v`` set
    for every ``v`` in ``1..n_numbers``.

    Raises ``ValueError`` if ``n_numbers`` is not a positive even integer or if
    some vertex has no partner (no perfect matching can exist).
    """
    if n_numbers < 2 or n_numbers % 2 != 0:
        raise ValueError(f"n_numbers must be a positive even integer, got {n_numbers!r}")

    squares: list[int] = []
    s = 2
    while s * s <= 2 * n_numbers - 1:
        squares.append(s * s)
        s += 1

    adj = [0] * (n_numbers + 1)
    for i in range(1, n_numbers + 1):
        mask = 0
        for sq in squares:
            j = sq - i
            if 1 <= j <= n_numbers and j != i:
                mask |= 1 << j
        adj[i] = mask

    full_mask = 0
    for v in range(1, n_numbers + 1):
        if adj[v] == 0:
            raise ValueError(
                f"vertex {v} has no square-sum partner in 1..{n_numbers}; no perfect matching exists"
            )
        full_mask |= 1 << v
    return adj, full_mask


def _components(adj: list[int], free: int) -> list[int]:
    """Split the subgraph induced on ``free`` into connected components (as masks)."""
    comps: list[int] = []
    remaining = free
    while remaining:
        comp = 0
        frontier = remaining & -remaining
        while frontier:
            comp |= frontier
            reach = 0
            f = frontier
            while f:
                b = f & -f
                f ^= b
                reach |= adj[b.bit_length() - 1]
            frontier = reach & free & ~comp
        comps.append(comp)
        remaining &= ~comp
    return comps


class _CapExceeded(Exception):
    """Raised internally when a component materialises too many solutions."""


class _CappedList:
    """emit target that collects into a list and aborts past ``limit`` items."""

    __slots__ = ("items", "_limit")

    def __init__(self, limit: int) -> None:
        self.items: list[Solution] = []
        self._limit = limit

    def __call__(self, sol: Solution) -> None:
        self.items.append(sol)
        if len(self.items) > self._limit:
            raise _CapExceeded


def _materialise_components(
    adj: list[int], comps: list[int], split: bool
) -> list[list[Solution]] | None:
    """Enumerate each component fully. Returns one solution list per component, or
    ``None`` if any component has no perfect matching. Raises ``_CapExceeded`` if a
    component overflows ``_SPLIT_MATERIALISE_CAP``."""
    per_comp: list[list[Solution]] = []
    for c in comps:
        sink = _CappedList(_SPLIT_MATERIALISE_CAP)
        _rec(adj, c, [], sink, split)
        if not sink.items:
            return None
        per_comp.append(sink.items)
    return per_comp


def _rec(adj: list[int], free: int, stack: Solution, emit: Emit, split: bool) -> None:
    """Enumerate every perfect matching of ``free``; call ``emit`` with a copy of
    ``stack`` for each. ``stack`` is restored before returning."""
    forced = 0  # pairs this frame pushed onto `stack`

    # --- forced-move propagation --------------------------------------------
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
            if forced:
                del stack[len(stack) - forced :]
            return
        if deg1_v < 0:
            break  # no forced move; min_v / min_deg reflect a full scan
        free ^= (1 << deg1_v) | (1 << deg1_p)
        stack.append((deg1_v, deg1_p) if deg1_v < deg1_p else (deg1_p, deg1_v))
        forced += 1

    if free == 0:
        emit(stack.copy())
        if forced:
            del stack[len(stack) - forced :]
        return

    if _popcount(free) & 1:  # odd set -> no perfect matching
        if forced:
            del stack[len(stack) - forced :]
        return

    # --- connected-component split ----------------------------------------
    # A low-degree pivot is the cheap signal that the free set may have
    # fragmented; only then is the O(V + E) component scan worth running.
    if split and min_deg <= 2 and _popcount(free) <= _SPLIT_THRESHOLD:
        comps = _components(adj, free)
        if len(comps) > 1:
            try:
                per_comp = _materialise_components(adj, comps, split)
            except _CapExceeded:
                pass  # fall through to plain MRV recursion below
            else:
                if per_comp is None:  # a component is unmatchable -> dead branch
                    if forced:
                        del stack[len(stack) - forced :]
                    return
                base = len(stack)
                for combo in itertools.product(*per_comp):
                    for part in combo:
                        stack.extend(part)
                    emit(stack.copy())
                    del stack[base:]
                if forced:
                    del stack[len(stack) - forced :]
                return

    # --- MRV branch ------------------------------------------------------
    bit_v = 1 << min_v
    m = adj[min_v] & free
    while m:
        b = m & -m
        m ^= b
        p = b.bit_length() - 1
        stack.append((min_v, p) if min_v < p else (p, min_v))
        _rec(adj, free ^ bit_v ^ b, stack, emit, split)
        stack.pop()

    if forced:
        del stack[len(stack) - forced :]


def _ensure_recursion_limit(n_numbers: int) -> None:
    needed = 4 * n_numbers + 1000
    if sys.getrecursionlimit() < needed:
        sys.setrecursionlimit(needed)


def enumerate_matchings(adj: list[int], free_mask: int, emit: Emit, *, split: bool = False) -> None:
    """Call ``emit`` once per perfect matching of the graph restricted to ``free_mask``."""
    _rec(adj, free_mask, [], emit, split)


def solve_prefix(
    adj: list[int],
    full_mask: int,
    prefix: tuple[Pair, ...] | list[Pair],
    emit: Emit,
    *,
    split: bool = False,
) -> None:
    """Enumerate every completion of the partial matching ``prefix`` (disjoint pairs
    drawn from ``full_mask``)."""
    free = full_mask
    stack: Solution = []
    for a, b in prefix:
        bit_a, bit_b = 1 << a, 1 << b
        if not (free & bit_a) or not (free & bit_b):
            raise ValueError(f"prefix pair ({a}, {b}) overlaps another pair or is out of range")
        free ^= bit_a | bit_b
        stack.append((a, b) if a < b else (b, a))
    _rec(adj, free, stack, emit, split)


def count_solutions(n_numbers: int, *, split: bool = False) -> int:
    """Number of square-sum pair partitions of ``1..n_numbers`` (OEIS A252897)."""
    _ensure_recursion_limit(n_numbers)
    adj, full = build_adjacency(n_numbers)
    total = 0

    def bump(_: Solution) -> None:
        nonlocal total
        total += 1

    _rec(adj, full, [], bump, split)
    return total


def iter_solutions(n_numbers: int, *, split: bool = False) -> Iterator[Solution]:
    """Yield every square-sum pair partition of ``1..n_numbers``.

    The recursion runs on a helper thread feeding a bounded queue, so consumer-side
    memory stays O(queue size + recursion depth) regardless of the solution count.
    """
    _ensure_recursion_limit(n_numbers)
    adj, full = build_adjacency(n_numbers)

    q: queue.Queue = queue.Queue(maxsize=512)
    sentinel = object()
    error: list[BaseException] = []

    def worker() -> None:
        try:
            _rec(adj, full, [], q.put, split)
        except BaseException as exc:  # surfaced to the consumer
            error.append(exc)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item
    t.join()
    if error:
        raise error[0]
