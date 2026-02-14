import pandas as pd


def write_to_csv(
    all_sets,
    output_csv,
):

    max_pairs = max(len(solution) for solution in all_sets)
    columns = ["Pair No"] + [f"Pair {i}" for i in range(1, max_pairs + 1)]
    rows = []
    for i, solution in enumerate(all_sets, 1):
        row = {"Pair No": i}
        for idx, (a, b) in enumerate(solution, 1):
            row[f"Pair {idx}"] = f"{a}-{b}"
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"\nAll complete sets have been written to '{output_csv}'.")


if __name__ == "__main__":
    # Example usage
    all_sets = [
        [(1, 2), (3, 4)],
        [(5, 6)],
        [(7, 8), (9, 10), (11, 12)],
    ]
    write_to_csv(all_sets, output_csv="data/complete_sets.csv")
