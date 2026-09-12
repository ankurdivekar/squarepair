"""CSV I/O for square-sum pair solutions.

All files share one format: a header ``Pair No,Pair 1,...,Pair K`` followed by one
row per solution, each cell the string ``"a-b"`` with ``a < b``. ``K`` is the pair
count (``n_numbers // 2``).

* ``write_to_csv`` -- in-memory list of solutions -> CSV (used from the notebook).
* ``StreamingCsvWriter`` -- append rows one at a time, constant memory.
* ``merge_shards`` -- concatenate headerless shard files into one final CSV.
* ``read_solutions`` -- stream solutions back out of a CSV written above.
"""

from __future__ import annotations

import csv
from typing import Iterable, Iterator, Sequence

Pair = tuple[int, int]
Solution = list[Pair]


def format_cells(pairs: Iterable[Pair]) -> list[str]:
    """Render pairs as ``"min-max"`` cell strings."""
    return [f"{a}-{b}" if a < b else f"{b}-{a}" for a, b in pairs]


def parse_cell(cell: str) -> Pair:
    """Inverse of ``format_cells``: ``"a-b"`` -> ``(a, b)``."""
    a, b = cell.split("-")
    return int(a), int(b)


def _header(n_pairs: int) -> list[str]:
    return ["Pair No"] + [f"Pair {i}" for i in range(1, n_pairs + 1)]


class StreamingCsvWriter:
    """Write solution rows incrementally. Use as a context manager::

        with StreamingCsvWriter("out.csv", n_pairs=21) as w:
            for sol in solutions:
                w.write_row(sol)
    """

    def __init__(self, output_csv: str, n_pairs: int) -> None:
        self._n_pairs = n_pairs
        self._fh = open(output_csv, "w", newline="")
        self._writer = csv.writer(self._fh, lineterminator="\n")
        self._writer.writerow(_header(n_pairs))
        self._count = 0

    def write_row(self, pairs: Iterable[Pair]) -> None:
        self._count += 1
        self._writer.writerow([self._count, *format_cells(pairs)])

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "StreamingCsvWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_to_csv(all_sets: Sequence[Sequence[Pair]], output_csv: str) -> None:
    """Write an in-memory sequence of solutions to ``output_csv``."""
    n_pairs = max((len(solution) for solution in all_sets), default=0)
    with open(output_csv, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(_header(n_pairs))
        for i, solution in enumerate(all_sets, 1):
            writer.writerow([i, *format_cells(solution)])
    print(f"\nAll complete sets have been written to '{output_csv}'.")


def merge_shards(shard_paths: Iterable[str], output_csv: str, n_pairs: int) -> int:
    """Concatenate headerless shard CSVs (in the given order) into ``output_csv``,
    prepending the header and assigning ``Pair No`` sequentially. Constant memory.
    Returns the total row count.
    """
    header = ",".join(_header(n_pairs)) + "\n"
    total = 0
    with open(output_csv, "w", newline="") as out:
        out.write(header)
        for path in shard_paths:
            with open(path, "r", newline="") as shard:
                for line in shard:
                    if not line.strip():
                        continue
                    total += 1
                    out.write(f"{total},{line}" if line.endswith("\n") else f"{total},{line}\n")
    return total


def read_solutions(csv_path: str, *, limit: int | None = None) -> Iterator[Solution]:
    """Stream solutions back out of a CSV written by ``write_to_csv`` or
    ``StreamingCsvWriter``, in row order.
    """
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield [parse_cell(cell) for cell in row[1:] if cell]


if __name__ == "__main__":
    demo = [
        [(1, 2), (3, 4)],
        [(5, 6)],
        [(7, 8), (9, 10), (11, 12)],
    ]
    write_to_csv(demo, output_csv="data/complete_sets.csv")
