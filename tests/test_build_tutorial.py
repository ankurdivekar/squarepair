"""The tutorial bakes in a solution-count table (`tutorial/index.html`). Keep it
honest: every entry must match the solver, and the page must stay self-contained
so it runs straight from ``file://`` after a clone.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.build_tutorial import BLOCK
from src.pair_solver import count_solutions

TUTORIAL = Path(__file__).resolve().parent.parent / "tutorial" / "index.html"


def _baked_counts() -> list[tuple[int, int]]:
    html = TUTORIAL.read_text()
    m = BLOCK.search(html)
    assert m, "AUTO-GENERATED markers missing from tutorial/index.html"
    arr = re.search(r"const COUNTS = (\[.*?\]);", html[m.start():m.end()], re.DOTALL)
    assert arr, "COUNTS array not found"
    return [tuple(p) for p in ast.literal_eval(arr.group(1).replace("\n", " "))]


def test_baked_counts_match_solver():
    for n, c in _baked_counts():
        if n <= 44:  # keep fast; larger n in the slow test below
            assert count_solutions(n) == c, f"tutorial COUNTS wrong for n={n}"


def test_baked_table_shape():
    counts = _baked_counts()
    assert counts[0][0] == 8 and counts[-1][0] == 60
    assert [n for n, _ in counts] == list(range(8, 61, 2))


def test_tutorial_is_self_contained():
    html = TUTORIAL.read_text()
    assert html.lstrip().lower().startswith("<!doctype html>")
    low = html.lower()
    for bad in ("fetch(", "xmlhttprequest", " import(", "new worker"):
        assert bad not in low, f"tutorial should not use {bad!r} (must run from file://)"
    for m in re.finditer(r'<script[^>]*\ssrc="([^"]+)"', html):
        raise AssertionError(f"unexpected external script: {m.group(1)}")
    for m in re.finditer(r'<link[^>]*\shref="(https?://[^"]+)"', html):
        assert "fonts.googleapis.com" in m.group(1) or "fonts.gstatic.com" in m.group(1), \
            f"unexpected external stylesheet: {m.group(1)}"


@pytest.mark.slow
def test_baked_counts_match_solver_full():
    for n, c in _baked_counts():
        assert count_solutions(n) == c, f"tutorial COUNTS wrong for n={n}"
