"""Render one circular-spline image per square-sum pair solution.

``draw_splines_on_circular_numbers`` draws a single solution. ``generate_images_for_n``
is the notebook-facing driver: it reads solutions for a given ``n`` from the ordered
CSV (falling back to the unordered complete-sets CSV) and renders one PNG per row.
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import Normalize

from src.csv_writer import Pair, read_solutions


def draw_splines_on_circular_numbers(n, number_pairs, output_file="circular_with_splines.png"):
    """
    Generate an image with N numbers arranged in a circular layout and draw splines between pairs.

    Parameters:
    -----------
    n : int
        Number of elements to arrange in a circle
    number_pairs : list of tuples
        List of (num1, num2) pairs to connect with splines
    output_file : str
        Path to save the output image
    """
    # Create figure with black background
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Circle radius around each number (fixed size for all numbers)
    circle_radius = 0.08

    # Calculate radius for text placement - SAME AS generate_circular_numbers function
    min_radius = (2.5 * circle_radius * n) / (2 * np.pi)
    radius = max(1.6, min_radius)

    # Adjust plot limits based on actual radius
    limit = radius + 0.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axis("off")

    # Store positions of each number for spline drawing
    positions = {}

    # Calculate angle for each number and store positions
    for i in range(n):
        angle = np.pi / 2 - (2 * np.pi * i / n)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        number = i + 1
        positions[number] = (x, y, angle)

    # Calculate color normalization based on sums of pairs
    if number_pairs:
        sums = [pair[0] + pair[1] for pair in number_pairs]
        min_sum = min(sums)
        max_sum = max(sums)

        # vmax is highest perfect square less than or equal to n*2 (max possible sum)
        # norm = Normalize(vmin=min_sum, vmax=max_sum)
        norm = Normalize(vmin=4, vmax=int(np.floor(np.sqrt(n * 2))) ** 2)

        colormap = cm.gist_rainbow

    # Draw splines first (so they appear behind the circles)
    for pair in number_pairs:
        num1, num2 = pair
        if num1 in positions and num2 in positions:
            x1, y1, angle1 = positions[num1]
            x2, y2, angle2 = positions[num2]

            # Calculate control points for smooth curve
            # Use points closer to center for control points
            control_factor = 0.4  # How much to pull towards center
            cx1 = x1 * control_factor
            cy1 = y1 * control_factor
            cx2 = x2 * control_factor
            cy2 = y2 * control_factor

            # Create smooth spline using Bezier-like curve
            points = np.array([[x1, y1], [cx1, cy1], [cx2, cy2], [x2, y2]])

            # Generate smooth curve
            t = np.linspace(0, 1, 100)
            # Cubic Bezier curve
            curve = (
                (1 - t)[:, np.newaxis] ** 3 * points[0]
                + 3 * (1 - t)[:, np.newaxis] ** 2 * t[:, np.newaxis] * points[1]
                + 3 * (1 - t)[:, np.newaxis] * t[:, np.newaxis] ** 2 * points[2]
                + t[:, np.newaxis] ** 3 * points[3]
            )

            # Determine color based on sum
            pair_sum = num1 + num2
            color = colormap(norm(pair_sum))

            # Draw the spline
            ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2, alpha=0.7)

    # Pre-calculate sequential rainbow colors for each circle
    circle_colors = cm.viridis(np.linspace(0, 1, n))

    # Draw circles and numbers on top of splines
    for i in range(n):
        angle = np.pi / 2 - (2 * np.pi * i / n)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        number = i + 1

        # Add circle around the number
        circle = patches.Circle(
            (x, y), circle_radius, fill=True, facecolor=circle_colors[i], edgecolor="white", linewidth=1.5, zorder=10
        )
        ax.add_patch(circle)

        # Choose text color for maximum visibility based on circle fill luminance
        r, g, b = circle_colors[i][:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "black" if luminance > 0.43 else "white"

        # Add text
        ax.text(
            x, y, str(number), color=text_color, fontsize=12, ha="center", va="center", fontweight="normal", zorder=11
        )

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close()

    print(f"Image with splines saved to {output_file}")


def generate_images_for_n(
    n: int,
    max_images: int,
    *,
    data_dir: str = "data",
    images_dir: str = "images",
) -> int:
    """Render up to ``max_images`` solutions for ``n`` into ``{images_dir}/n{n}/``.

    Reads ``{data_dir}/ordered_sets_n{n}.csv`` if it exists (solutions pre-sorted for
    minimal change between frames), otherwise ``{data_dir}/complete_sets_n{n}.csv``.
    The output directory is cleared first. Returns the number of images written.
    """
    ordered_csv = Path(data_dir) / f"ordered_sets_n{n}.csv"
    complete_csv = Path(data_dir) / f"complete_sets_n{n}.csv"
    csv_path = ordered_csv if ordered_csv.exists() else complete_csv

    out_dir = Path(images_dir) / f"n{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in out_dir.iterdir():
        if file.is_file():
            file.unlink()

    count = 0
    for pairs in read_solutions(str(csv_path), limit=max_images):
        draw_splines_on_circular_numbers(n, pairs, str(out_dir / f"{count:06}.png"))
        count += 1
    return count


if __name__ == "__main__":
    # Example: n=60 with a few connections
    pairs_60: list[Pair] = [(2, 45), (28, 82), (15, 50), (10, 55), (5, 35)]
    draw_splines_on_circular_numbers(60, pairs_60, "circular_60_splines.png")
