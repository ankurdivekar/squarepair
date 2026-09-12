"""Correctness tests for the square-sum pair enumerator.

Ground truth: OEIS A252897 a(k) with k = n_numbers // 2, cross-checked against the
committed CSVs in data/.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.pair_solver import (
    build_adjacency,
    count_solutions,
    iter_solutions,
    solve_prefix,
)
from src.run_enumeration import enumerate_to_csv, generate_prefixes

REPO = Path(__file__).resolve().parent.parent

# n_numbers -> number of complete sets  (A252897 a(n_numbers//2))
KNOWN = {8: 1, 16: 1, 24: 1, 26: 6, 28: 18, 32: 36, 34: 156, 42: 2603}


def _canon(sol):
    return frozenset(frozenset(p) for p in sol)


def _canon_set(sols):
    return {_canon(s) for s in sols}


@pytest.mark.parametrize("n,expected", sorted(KNOWN.items()))
def test_count_matches_known(n, expected):
    assert count_solutions(n) == expected


@pytest.mark.parametrize("n", [26, 28, 34, 42])
def test_no_duplicates_or_omissions(n):
    sols = list(iter_solutions(n))
    canon = _canon_set(sols)
    assert len(sols) == len(canon) == KNOWN[n]
    # every pair sums to a perfect square, every vertex used once
    for s in sols:
        used = sorted(x for p in s for x in p)
        assert used == list(range(1, n + 1))
        for a, b in s:
            r = round((a + b) ** 0.5)
            assert r * r == a + b


@pytest.mark.parametrize("n", [28, 34])
def test_split_equivalence(n):
    plain = _canon_set(iter_solutions(n, split=False))
    with_split = _canon_set(iter_solutions(n, split=True))
    assert plain and plain == with_split


@pytest.mark.parametrize("n", [28, 34])
def test_prefix_partition(n):
    adj, full = build_adjacency(n)
    prefixes = generate_prefixes(adj, full, target_tasks=32)
    seen_masks = []
    union: set = set()
    total = 0
    for pf in prefixes:
        got: list = []
        solve_prefix(adj, full, pf, got.append)
        total += len(got)
        cset = _canon_set(got)
        assert union.isdisjoint(cset), "prefixes overlap"
        union |= cset
        seen_masks.append(pf)
    assert total == len(union) == KNOWN[n]


def test_csv_golden_n42(tmp_path):
    out = tmp_path / "n42.csv"
    total = enumerate_to_csv(42, str(out), workers=2, shard_dir=str(tmp_path / "sh"))
    assert total == 2603

    with out.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == ["Pair No"] + [f"Pair {i}" for i in range(1, 22)]
        rows = list(reader)
    assert len(rows) == 2603
    assert [int(r[0]) for r in rows] == list(range(1, 2604))
    assert all(len(r) == 22 for r in rows)

    def parse(path):
        with open(path) as fh:
            r = csv.reader(fh)
            next(r)
            return {
                frozenset(frozenset(map(int, c.split("-"))) for c in row[1:] if c)
                for row in r
            }

    assert parse(out) == parse(REPO / "data" / "complete_sets_n42.csv")


def test_odd_and_isolated_raise():
    with pytest.raises(ValueError):
        build_adjacency(27)  # odd
    with pytest.raises(ValueError):
        build_adjacency(2)  # 1+2=3 not square -> vertex isolated


@pytest.mark.slow
def test_n60_count():
    assert count_solutions(60) == 4366714
