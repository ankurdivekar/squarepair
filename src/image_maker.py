import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import Normalize

# # Calculate all perfect squares including and between a min and max value
# def perfect_squares_in_range(min_value: int, max_value: int) -> list[int]:
#     squares = []
#     n = 1
#     while True:
#         square = n * n
#         if square > max_value:
#             break
#         if square >= min_value:
#             squares.append(square)
#         n += 1
#     return squares


# # Create colormap with a specified number of discrete colors. Maximize contrast by using a colormap that has good differentiation between colors (e.g., Set3, tab20, or viridis).
# def create_colormap(num_colors: int):
#     # Use a colormap with good contrast for discrete colors
#     base_colormap = cm.get_cmap(
#         "Set3"
#     )  # You can change to other colormaps: viridis, plasma, inferno, coolwarm, rainbow

#     # Create a new colormap with the specified number of discrete colors
#     colors = base_colormap(np.linspace(0, 1, num_colors))
#     return colors


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

    # Draw circles and numbers on top of splines
    for i in range(n):
        angle = np.pi / 2 - (2 * np.pi * i / n)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        number = i + 1

        # Add circle around the number
        circle = patches.Circle(
            (x, y), circle_radius, fill=True, facecolor="black", edgecolor="white", linewidth=1.5, zorder=10
        )
        ax.add_patch(circle)

        # Add text
        ax.text(x, y, str(number), color="white", fontsize=12, ha="center", va="center", fontweight="normal", zorder=11)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close()

    print(f"Image with splines saved to {output_file}")


if __name__ == "__main__":
    # Example 1: n=60 with a few connections
    pairs_60 = [(2, 45), (28, 82), (15, 50), (10, 55), (5, 35)]
    draw_splines_on_circular_numbers(60, pairs_60, "circular_60_splines.png")

    # # Example 2: n=100 with connections
    # pairs_100 = [(2, 45), (28, 82), (15, 90), (50, 95), (10, 70), (20, 80)]
    # draw_splines_on_circular_numbers(100, pairs_100, "circular_100_splines.png")

    # # Example 3: n=24 with many connections (demonstrates color gradient)
    # pairs_24 = [(1, 12), (2, 11), (3, 10), (5, 20), (8, 18), (15, 24)]
    # draw_splines_on_circular_numbers(24, pairs_24, "circular_24_splines.png")
