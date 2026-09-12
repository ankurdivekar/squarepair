# squarepair

Finding and visualizing pairs of numbers &lt; 2n that add up to perfect squares.

For `1..N` (N even) a "complete set" partitions the numbers into `N/2` pairs whose
sums are all perfect squares — a perfect matching of the *square-sum graph*. The
number of complete sets as a function of `N/2` is [OEIS A252897](https://oeis.org/A252897).

## Enumerating complete sets

The solver (`src/pair_solver.py`) is bitmask backtracking with forced-move
propagation and MRV branching. `src/run_enumeration.py` is the parallel,
stream-to-CSV driver.

```bash
# full enumeration -> data/complete_sets_n60.csv  (CPython, all cores)
uv run python -m src.run_enumeration --n 60

# just the count (OEIS A252897 term), no CSV
uv run python -m src.run_enumeration --n 60 --count-only
```

From the notebook, `generate.ipynb` cell "Generate complete sets…" calls
`generate_complete_sets_to_csv(n_numbers=n, output_csv=…)`.

### Large N — run under PyPy

The search is pure-Python and PyPy-friendly; PyPy gives a large speedup for free.

```bash
uv python install pypy3.11
uv run --python pypy3.11 -m src.run_enumeration --n 100 \
    --out data/complete_sets_n100.csv \
    --workers 16 --shard-dir /path/to/scratch/n100
```

`--shard-dir` keeps the per-worker shard files and a `manifest.json`; a re-run with
the same `--shard-dir` resumes (skips finished shards). Drop `--shard-dir` for a
one-shot run into a temp dir that is cleaned up afterwards.

The downstream image/video pipeline (`generate.ipynb`) still runs on the CPython
`.venv` — only the solver benefits from PyPy.

## Interactive tutorial

`tutorial/index.html` is a self-contained, high-school-level walkthrough of the
problem and the search: the graph model, why brute force is hopeless, a
step-by-step player for the backtracking search, and a gallery of solutions. It
needs no build step and no server — open it straight from a clone:

```bash
open tutorial/index.html          # macOS  (or: xdg-open / just double-click)
```

Everything is computed in the browser except the solution-count table for large
`n`, which is pre-computed by the repo's own solver. To regenerate it (e.g. after
changing the solver, or to extend the table):

```bash
uv run python -m src.build_tutorial          # n = 8..60
uv run python -m src.build_tutorial --max 72  # extend (slow past ~62)
```

## Tests

```bash
uv run pytest -m "not slow"    # fast correctness checks
uv run pytest                  # includes the n=60 count (~30s on CPython)
```
