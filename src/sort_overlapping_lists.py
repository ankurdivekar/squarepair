"""
Sort lists of number pairs to maximize overlap between consecutive lists.
"""

import random
from itertools import permutations
from typing import List, Tuple


def count_common_pairs(list1: List[Tuple[int, int]], list2: List[Tuple[int, int]]) -> int:
    """
    Count the number of common pairs between two lists.
    Considers (a, b) and (b, a) as the same pair.

    Parameters
    ----------
    list1, list2 : List of tuples
        Lists of number pairs to compare

    Returns
    -------
    int
        Number of common pairs

    Examples
    --------
    >>> count_common_pairs([(1, 2), (3, 4)], [(2, 1), (5, 6)])
    1  # (1,2) matches (2,1)
    """
    # Normalize pairs so (a, b) and (b, a) are treated the same
    set1 = {tuple(sorted(pair)) for pair in list1}
    set2 = {tuple(sorted(pair)) for pair in list2}
    return len(set1 & set2)


def calculate_total_overlap(lists: List[List[Tuple[int, int]]]) -> int:
    """
    Calculate total overlap between consecutive lists.

    Parameters
    ----------
    lists : List of lists of tuples
        Ordered sequence of pair lists

    Returns
    -------
    int
        Total number of common pairs across all consecutive pairs
    """
    total = 0
    for i in range(len(lists) - 1):
        total += count_common_pairs(lists[i], lists[i + 1])
    return total


def sort_by_greedy(lists: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    Sort lists using a greedy algorithm.
    Start with any list, then repeatedly pick the unused list with maximum overlap.

    Fast but not guaranteed optimal. Good for large numbers of lists (100+).

    Parameters
    ----------
    lists : List of lists of tuples
        Lists to sort

    Returns
    -------
    List of lists
        Sorted lists maximizing consecutive overlap

    Examples
    --------
    >>> lists = [[(1,2), (3,4)], [(2,1), (5,6)], [(5,6), (7,8)]]
    >>> sorted_lists = sort_by_greedy(lists)
    """
    if len(lists) <= 1:
        return lists.copy()

    result = [lists[0]]
    remaining = set(range(1, len(lists)))

    while remaining:
        current = result[-1]
        best_idx = None
        best_overlap = -1

        # Find the remaining list with maximum overlap
        for idx in remaining:
            overlap = count_common_pairs(current, lists[idx])
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        result.append(lists[best_idx])
        remaining.remove(best_idx)

    return result


def sort_by_greedy_best_start(lists: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    Try greedy algorithm starting from each list, return the best result.

    More thorough than simple greedy, still fast for moderate sizes (< 1000 lists).

    Parameters
    ----------
    lists : List of lists of tuples
        Lists to sort

    Returns
    -------
    List of lists
        Best sorted sequence found
    """
    if len(lists) <= 1:
        return lists.copy()

    best_result = None
    best_overlap = -1

    # Try starting from each list
    for start_idx in range(len(lists)):
        result = [lists[start_idx]]
        remaining = set(range(len(lists)))
        remaining.remove(start_idx)

        while remaining:
            current = result[-1]
            best_next = None
            best_next_overlap = -1

            for idx in remaining:
                overlap = count_common_pairs(current, lists[idx])
                if overlap > best_next_overlap:
                    best_next_overlap = overlap
                    best_next = idx

            result.append(lists[best_next])
            remaining.remove(best_next)

        total_overlap = calculate_total_overlap(result)
        if total_overlap > best_overlap:
            best_overlap = total_overlap
            best_result = result

    return best_result


def sort_by_simulated_annealing(
    lists: List[List[Tuple[int, int]]],
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    iterations: int = 10000,
) -> List[List[Tuple[int, int]]]:
    """
    Sort lists using simulated annealing optimization.

    Best quality results, but slower. Good for < 100 lists.

    Parameters
    ----------
    lists : List of lists of tuples
        Lists to sort
    initial_temp : float
        Starting temperature for annealing
    cooling_rate : float
        Temperature reduction per iteration (0 < rate < 1)
    iterations : int
        Number of iterations to run

    Returns
    -------
    List of lists
        Optimized sorted sequence
    """
    if len(lists) <= 1:
        return lists.copy()

    # Start with greedy solution
    current = sort_by_greedy(lists)
    current_score = calculate_total_overlap(current)
    best = current.copy()
    best_score = current_score

    temp = initial_temp

    for iteration in range(iterations):
        # Generate neighbor by swapping two random positions
        i, j = random.sample(range(len(current)), 2)
        neighbor = current.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        neighbor_score = calculate_total_overlap(neighbor)

        # Accept if better, or probabilistically if worse
        delta = neighbor_score - current_score
        if delta > 0 or random.random() < pow(2.71828, delta / temp):
            current = neighbor
            current_score = neighbor_score

            if current_score > best_score:
                best = current.copy()
                best_score = current_score

        # Cool down
        temp *= cooling_rate

    return best


def sort_by_exhaustive(lists: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    Try all possible orderings and return the best.

    ONLY use for very small lists (< 10 items). Factorial time complexity!

    Parameters
    ----------
    lists : List of lists of tuples
        Lists to sort (should be < 10 items)

    Returns
    -------
    List of lists
        Optimal sorted sequence
    """
    if len(lists) > 10:
        raise ValueError("Exhaustive search only practical for < 10 lists. Use greedy or simulated annealing instead.")

    if len(lists) <= 1:
        return lists.copy()

    best_order = lists
    best_overlap = calculate_total_overlap(lists)

    for perm in permutations(lists):
        overlap = calculate_total_overlap(list(perm))
        if overlap > best_overlap:
            best_overlap = overlap
            best_order = list(perm)

    return best_order


def sort_lists_smart(lists: List[List[Tuple[int, int]]], method: str = "auto") -> List[List[Tuple[int, int]]]:
    """
    Automatically choose the best sorting method based on list size.

    Parameters
    ----------
    lists : List of lists of tuples
        Lists to sort
    method : str, default='auto'
        Method to use: 'auto', 'greedy', 'greedy_best', 'annealing', or 'exhaustive'

    Returns
    -------
    List of lists
        Sorted lists

    Examples
    --------
    >>> lists = [[(1,2), (3,4)], [(2,1), (5,6)], [(5,6), (7,8)]]
    >>> sorted_lists = sort_lists_smart(lists)
    >>> # Now consecutive lists have maximum overlap
    """
    n = len(lists)

    if method == "auto":
        if n <= 8:
            method = "exhaustive"
        elif n <= 50:
            method = "annealing"
        elif n <= 200:
            method = "greedy_best"
        else:
            method = "greedy"

    print(f"Sorting {n} lists using method: {method}")

    if method == "exhaustive":
        return sort_by_exhaustive(lists)
    elif method == "greedy":
        return sort_by_greedy(lists)
    elif method == "greedy_best":
        return sort_by_greedy_best_start(lists)
    elif method == "annealing":
        return sort_by_simulated_annealing(lists)
    else:
        raise ValueError(f"Unknown method: {method}")


# Example usage and testing
if __name__ == "__main__":
    # Example: Create some test lists
    test_lists = [
        [(1, 2), (3, 4), (5, 6)],
        [(2, 1), (7, 8)],  # shares (1,2) with first
        [(5, 6), (9, 10)],  # shares (5,6) with first
        [(7, 8), (11, 12)],  # shares (7,8) with second
        [(9, 10), (13, 14)],  # shares (9,10) with third
    ]

    print("Original order:")
    print(f"Total overlap: {calculate_total_overlap(test_lists)}")
    for i, lst in enumerate(test_lists):
        print(f"  {i}: {lst}")

    print("\n" + "=" * 60)
    sorted_lists = sort_lists_smart(test_lists)

    print("\nOptimized order:")
    print(f"Total overlap: {calculate_total_overlap(sorted_lists)}")
    for i, lst in enumerate(sorted_lists):
        if i > 0:
            overlap = count_common_pairs(sorted_lists[i - 1], lst)
            print(f"  {i}: {lst} (overlap with prev: {overlap})")
        else:
            print(f"  {i}: {lst}")
