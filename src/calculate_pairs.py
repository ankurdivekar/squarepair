import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional


def get_square_combinations(n_numbers: int) -> list[tuple[int, int]]:
    """Find all pairs of numbers that sum to a perfect square.
    Optimized: O(n²) using set for pair tracking and math.isqrt."""
    squares: list[tuple[int, int]] = []
    pairs: set[tuple[int, int]] = set()

    for i in range(1, n_numbers + 1):
        for j in range(1, n_numbers + 1):
            if i != j and (j, i) not in pairs:
                pairs.add((i, j))

                # check if sum is perfect square
                s = i + j
                sqrt_s = math.isqrt(s)
                if sqrt_s * sqrt_s == s:
                    squares.append((i, j))
    return squares


def explore_branch(
    first_num: int,
    first_partner: int,
    adjacency_dict: dict[int, set[int]],
    total_numbers: int,
    worker_id: int,
    initial_pairs: list[tuple[int, int]] | None = None,
) -> list[list[tuple[int, int]]]:
    """Worker function: explore all solutions starting with initial pairs.
    If initial_pairs is provided, starts with those pairs already made."""
    used: set[int] = {first_num, first_partner}
    pairs: list[tuple[int, int]] = [(first_num, first_partner)]

    # Add any pre-computed initial pairs
    if initial_pairs:
        for num1, num2 in initial_pairs:
            used.add(num1)
            used.add(num2)
            pairs.append((num1, num2))

    branch_solutions: list[list[tuple[int, int]]] = []
    attempts = 0

    def backtrack() -> None:
        nonlocal attempts
        attempts += 1

        # Success: all numbers are paired
        if len(used) == total_numbers:
            branch_solutions.append(list(pairs))
            if len(branch_solutions) % 100 == 0:
                print(f"  Worker {worker_id}: Found {len(branch_solutions)} solutions (attempts: {attempts:,})")
            return

        # Find the first unused number to pair
        start_num: Optional[int] = None
        for num in range(1, total_numbers + 1):
            if num not in used:
                start_num = num
                break

        if start_num is None:
            return

        # Try each valid partner for start_num
        for partner in adjacency_dict[start_num]:
            if partner not in used:
                used.add(start_num)
                used.add(partner)
                pairs.append((start_num, partner))

                backtrack()

                used.remove(start_num)
                used.remove(partner)
                pairs.pop()

    backtrack()
    print(f"✓ Worker {worker_id} finished: {len(branch_solutions)} solutions, {attempts:,} attempts")
    return branch_solutions


def explore_multiple_branches(
    first_num: int,
    partners: list[int],
    adjacency_dict: dict[int, set[int]],
    total_numbers: int,
    worker_id: int,
    initial_pairs: list[tuple[int, int]] | None = None,
) -> list[list[tuple[int, int]]]:
    """Worker function: explore multiple branches sequentially."""
    all_branch_solutions: list[list[tuple[int, int]]] = []

    for partner in partners:
        branch_solutions = explore_branch(first_num, partner, adjacency_dict, total_numbers, worker_id, initial_pairs)
        all_branch_solutions.extend(branch_solutions)

    return all_branch_solutions


def generate_level2_branches(
    adjacency_dict: dict[int, set[int]], n_numbers: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Generate all valid 2-level branch configurations: (pair1, pair2)."""
    first_neighbors = sorted(adjacency_dict[1])
    level2_branches = []

    for first_partner in first_neighbors:
        # After fixing (1, first_partner), find next unpaired number
        used = {1, first_partner}
        next_num = None
        for num in range(2, n_numbers + 1):
            if num not in used:
                next_num = num
                break

        if next_num and next_num in adjacency_dict:
            for second_partner in adjacency_dict[next_num]:
                if second_partner not in used:
                    level2_branches.append(((1, first_partner), (next_num, second_partner)))

    return level2_branches


def generate_level3_branches(
    adjacency_dict: dict[int, set[int]], n_numbers: int
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Generate all valid 3-level branch configurations: (pair1, pair2, pair3)."""
    level2_branches = generate_level2_branches(adjacency_dict, n_numbers)
    level3_branches = []

    for pair1, pair2 in level2_branches:
        # After fixing pair1 and pair2, find next unpaired number
        used = {pair1[0], pair1[1], pair2[0], pair2[1]}
        next_num = None
        for num in range(1, n_numbers + 1):
            if num not in used:
                next_num = num
                break

        if next_num and next_num in adjacency_dict:
            for third_partner in adjacency_dict[next_num]:
                if third_partner not in used:
                    level3_branches.append((pair1, pair2, (next_num, third_partner)))

    return level3_branches


def decide_splitting_strategy(num_cpus: int, n_numbers: int, adjacency_dict: dict[int, set[int]]) -> tuple[int, int]:
    """Decide splitting depth based on CPU count.
    Returns: (splitting_depth, estimated_branches)"""
    first_neighbors = len(adjacency_dict[1])
    level1_branches = first_neighbors
    level2_branches = len(generate_level2_branches(adjacency_dict, n_numbers))

    # Strategy: Choose depth that generates enough branches to use all cores
    # but not excessively (aim for 1.5x to 2x cores for some load balancing)
    target_branches = num_cpus * 1.5

    if level1_branches >= target_branches:
        return 1, level1_branches
    elif level2_branches >= target_branches:
        return 2, level2_branches
    else:
        # For very high core counts, use level 3
        level3_branches = len(generate_level3_branches(adjacency_dict, n_numbers))
        return 3, level3_branches


def find_all_complete_sets_parallel(adjacency: dict[int, set[int]], n_numbers: int) -> list[list[tuple[int, int]]]:
    """Parallel backtracking with adaptive branch splitting.
    Automatically chooses splitting depth based on available CPU cores."""
    num_cpus = os.cpu_count() or 1
    splitting_depth, estimated_branches = decide_splitting_strategy(num_cpus, n_numbers, adjacency)

    print(f"\n{'=' * 60}")
    print("Adaptive Parallelization Strategy")
    print(f"{'=' * 60}")
    print(f"Available CPU cores: {num_cpus}")
    print(f"Splitting depth chosen: Level {splitting_depth}")
    print(f"Estimated branches: {estimated_branches}")
    print(f"Target branches: ~{num_cpus * 1.5:.0f}")
    print(f"{'=' * 60}\n")

    if splitting_depth == 1:
        return _parallel_level1(adjacency, n_numbers, num_cpus)
    elif splitting_depth == 2:
        return _parallel_level2(adjacency, n_numbers, num_cpus)
    else:
        return _parallel_level3(adjacency, n_numbers, num_cpus)


def _parallel_level1(adjacency: dict[int, set[int]], n_numbers: int, num_cpus: int) -> list[list[tuple[int, int]]]:
    """Level 1 splitting: split on (1, neighbors)."""
    first_neighbors = sorted(adjacency[1])
    num_branches = len(first_neighbors)
    num_workers = min(num_branches, num_cpus)

    print("Strategy: Level 1 Splitting")
    print(f"  - First number 1 has {num_branches} neighbors")
    print(f"  - Creating {num_workers} workers (1 branch per worker)")
    print("  - Each worker explores: (1, neighbor) → full search\n")

    if num_branches <= num_workers:
        # Each branch gets its own worker
        print(f"Starting parallel search with {num_workers} workers...\n")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    explore_branch,
                    1,
                    neighbor,
                    adjacency,
                    n_numbers,
                    i + 1,
                )
                for i, neighbor in enumerate(first_neighbors)
            ]

            all_solutions: list[list[tuple[int, int]]] = []
            for future in futures:
                branch_solutions = future.result()
                all_solutions.extend(branch_solutions)
    else:
        # Distribute multiple branches per worker
        branches_per_worker = [[] for _ in range(num_workers)]
        for i, neighbor in enumerate(first_neighbors):
            branches_per_worker[i % num_workers].append(neighbor)

        print(f"Starting parallel search with {num_workers} workers (multiple branches per worker)...\n")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    explore_multiple_branches,
                    1,
                    branches,
                    adjacency,
                    n_numbers,
                    i + 1,
                )
                for i, branches in enumerate(branches_per_worker)
            ]

            all_solutions: list[list[tuple[int, int]]] = []
            for future in futures:
                branch_solutions = future.result()
                all_solutions.extend(branch_solutions)

    return all_solutions


def _parallel_level2(adjacency: dict[int, set[int]], n_numbers: int, num_cpus: int) -> list[list[tuple[int, int]]]:
    """Level 2 splitting: split on (pair1, pair2) combinations."""
    level2_branches = generate_level2_branches(adjacency, n_numbers)
    num_branches = len(level2_branches)
    num_workers = min(num_branches, num_cpus)

    print("Strategy: Level 2 Splitting")
    print(f"  - Generated {num_branches} 2-level branch configurations")
    print(f"  - Creating {num_workers} workers")
    print("  - Each worker explores: (1, p1), (n, p2) → full search\n")
    print(f"Starting parallel search with {num_workers} workers...\n")

    if num_branches <= num_workers:
        # Each branch gets its own worker
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    explore_branch,
                    branch[0][0],
                    branch[0][1],
                    adjacency,
                    n_numbers,
                    i + 1,
                    [branch[1]],
                )
                for i, branch in enumerate(level2_branches)
            ]

            all_solutions: list[list[tuple[int, int]]] = []
            for future in futures:
                branch_solutions = future.result()
                all_solutions.extend(branch_solutions)
    else:
        # Distribute multiple branches per worker
        branches_per_worker = [[] for _ in range(num_workers)]
        for i, branch in enumerate(level2_branches):
            branches_per_worker[i % num_workers].append(branch)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_explore_multiple_level2_branches, branches, adjacency, n_numbers, i + 1)
                for i, branches in enumerate(branches_per_worker)
            ]

            all_solutions: list[list[tuple[int, int]]] = []
            for future in futures:
                branch_solutions = future.result()
                all_solutions.extend(branch_solutions)

    return all_solutions


def _parallel_level3(adjacency: dict[int, set[int]], n_numbers: int, num_cpus: int) -> list[list[tuple[int, int]]]:
    """Level 3 splitting: split on (pair1, pair2, pair3) combinations."""
    level3_branches = generate_level3_branches(adjacency, n_numbers)
    num_branches = len(level3_branches)
    num_workers = min(num_branches, num_cpus)

    print("Strategy: Level 3 Splitting")
    print(f"  - Generated {num_branches} 3-level branch configurations")
    print(f"  - Creating {num_workers} workers")
    print("  - Each worker explores: (1, p1), (n, p2), (m, p3) → full search\n")
    print(f"Starting parallel search with {num_workers} workers...\n")

    if num_branches <= num_workers:
        # Each branch gets its own worker
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    explore_branch,
                    branch[0][0],
                    branch[0][1],
                    adjacency,
                    n_numbers,
                    i + 1,
                    [branch[1], branch[2]],
                )
                for i, branch in enumerate(level3_branches)
            ]

            all_solutions: list[list[tuple[int, int]]] = []
            for future in futures:
                branch_solutions = future.result()
                all_solutions.extend(branch_solutions)
    else:
        # Distribute multiple branches per worker
        branches_per_worker = [[] for _ in range(num_workers)]
        for i, branch in enumerate(level3_branches):
            branches_per_worker[i % num_workers].append(branch)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_explore_multiple_level3_branches, branches, adjacency, n_numbers, i + 1)
                for i, branches in enumerate(branches_per_worker)
            ]

            all_solutions: list[list[tuple[int, int]]] = []
            for future in futures:
                branch_solutions = future.result()
                all_solutions.extend(branch_solutions)

    return all_solutions


def _explore_multiple_level2_branches(
    branches: list[tuple[tuple[int, int], tuple[int, int]]],
    adjacency_dict: dict[int, set[int]],
    total_numbers: int,
    worker_id: int,
) -> list[list[tuple[int, int]]]:
    """Explore multiple level-2 branches sequentially in a single worker."""
    all_solutions: list[list[tuple[int, int]]] = []
    for pair1, pair2 in branches:
        solutions = explore_branch(pair1[0], pair1[1], adjacency_dict, total_numbers, worker_id, [pair2])
        all_solutions.extend(solutions)
    return all_solutions


def _explore_multiple_level3_branches(
    branches: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
    adjacency_dict: dict[int, set[int]],
    total_numbers: int,
    worker_id: int,
) -> list[list[tuple[int, int]]]:
    """Explore multiple level-3 branches sequentially in a single worker."""
    all_solutions: list[list[tuple[int, int]]] = []
    for pair1, pair2, pair3 in branches:
        solutions = explore_branch(pair1[0], pair1[1], adjacency_dict, total_numbers, worker_id, [pair2, pair3])
        all_solutions.extend(solutions)
    return all_solutions


def find_all_complete_sets_backtracking(adjacency: dict[int, set[int]], n_numbers: int) -> list[list[tuple[int, int]]]:
    """Use backtracking to find ALL valid complete sets.
    This builds solutions incrementally and prunes impossible branches."""
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    all_solutions: list[list[tuple[int, int]]] = []
    attempts: list[int] = [0]  # Track attempts for progress reporting

    def backtrack() -> None:
        attempts[0] += 1
        if attempts[0] % 10000000 == 0:
            print(f"Attempts: {attempts[0]:,}, Solutions found: {len(all_solutions)}, Current pairs: {len(pairs)}")

        # Success: all numbers are paired - save this solution and continue searching
        if len(used) == n_numbers:
            all_solutions.append(list(pairs))  # Make a copy of the current solution
            # if len(all_solutions) % 1000 == 0:
            #     print(f"  → Found solution #{len(all_solutions)}")
            return  # Continue backtracking to find more solutions

        # Find the first unused number to pair
        start_num: Optional[int] = None
        for num in range(1, n_numbers + 1):
            if num not in used:
                start_num = num
                break

        if start_num is None:
            return

        # Try each valid partner for start_num
        for partner in adjacency[start_num]:
            if partner not in used:
                # Make the pairing
                used.add(start_num)
                used.add(partner)
                pairs.append((start_num, partner))

                # Recurse
                backtrack()

                # Backtrack to explore other possibilities
                used.remove(start_num)
                used.remove(partner)
                pairs.pop()

    # Start the backtracking
    backtrack()
    return all_solutions


def generate_complete_sets(n_numbers: int) -> list[list[tuple[int, int]]]:
    """Generate all complete sets of square-sum pairs for numbers 1..n_numbers

    Args:
        n_numbers: Upper bound of the number range (1..n_numbers).

    Returns a list of solutions, where each solution is a list of (a, b) pairs
    such that every number 1..n_numbers is used exactly once and each a+b is a
    perfect square.
    """
    square_pairs: list[tuple[int, int]] = get_square_combinations(n_numbers)
    print(f"Found {len(square_pairs)} pairs of numbers that sum to a perfect square, between 1 and {n_numbers}.")

    # Check if every number from 1 to n_numbers is present in the squares list
    numbers_in_squares: set[int] = set()
    for i, j in square_pairs:
        numbers_in_squares.add(i)
        numbers_in_squares.add(j)

    if not len(numbers_in_squares) == n_numbers:
        raise BaseException(
            f"Not all numbers from 1 to {n_numbers} are present in the squares list. Missing: {set(range(1, n_numbers + 1)) - numbers_in_squares}"
        )

    # Build adjacency graph: number -> set of numbers it pairs with
    print("Building adjacency graph...")
    adjacency: dict[int, set[int]] = {i: set() for i in range(1, n_numbers + 1)}
    for i, j in square_pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)

    print(f"Graph built. Number 1 has {len(adjacency[1])} possible partners.\n")

    # Run parallel version
    start_time = time.time()
    all_sets: list[list[tuple[int, int]]] = find_all_complete_sets_parallel(adjacency, n_numbers)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Found {len(all_sets)} valid complete sets for n={n_numbers}")

    return all_sets


# Main execution
if __name__ == "__main__":
    n_numbers: int = 42
    output_csv: str = f"data/complete_sets_n{n_numbers}.csv"
    all_sets = generate_complete_sets(n_numbers, output_csv)

    print("\n" + 50 * "=")
    if all_sets:
        print(f"✓ Success! Found {len(all_sets)} valid complete sets for n={n_numbers}!")
    else:
        print("No valid complete sets found.")
    print("\n" + 50 * "=")
