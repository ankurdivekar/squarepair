"""
Video creation module for combining images into videos.
Easy to import and use from other Python scripts.
"""

from pathlib import Path
from typing import Callable, List, Optional, Union

import cv2


def create_video_from_images(
    image_folder: Union[str, Path],
    output_video: Union[str, Path],
    fps: int = 30,
    frames_per_image: int = 30,
    image_pattern: str = "*.png",
    sort_key: Optional[Callable] = None,
    verbose: bool = True,
) -> str:
    """
    Create a video from a sequence of images.

    Parameters
    ----------
    image_folder : str or Path
        Path to folder containing images
    output_video : str or Path
        Path for output video file (e.g., 'output.mp4')
    fps : int, default=30
        Frames per second for the video
    frames_per_image : int, default=30
        Number of frames each image persists on screen
        At 30 fps: 30 frames = 1 second, 60 frames = 2 seconds, etc.
    image_pattern : str, default='*.png'
        Glob pattern to match image files
    sort_key : callable, optional
        Function to sort images (default: alphabetical by filename)
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    str
        Path to the created video file

    Examples
    --------
    >>> create_video_from_images('my_images/', 'output.mp4', frames_per_image=60)

    >>> # With custom sorting
    >>> create_video_from_images(
    ...     'my_images/',
    ...     'output.mp4',
    ...     sort_key=lambda x: int(x.stem.split('_')[-1])
    ... )
    """
    image_folder = Path(image_folder)
    image_files = sorted(image_folder.glob(image_pattern), key=sort_key if sort_key else lambda x: x.name)

    if not image_files:
        raise ValueError(f"No images found in {image_folder} matching pattern {image_pattern}")

    if verbose:
        duration = len(image_files) * frames_per_image / fps
        print(f"Found {len(image_files)} images")
        print(f"Video settings: {fps} FPS, {frames_per_image} frames per image")
        print(f"Each image will be shown for {frames_per_image / fps:.2f} seconds")
        print(f"Total video duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")

    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        raise ValueError(f"Could not read image: {image_files[0]}")

    height, width, channels = first_image.shape
    if verbose:
        print(f"Video dimensions: {width}x{height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("Could not open video writer")

    # Process each image
    for idx, image_path in enumerate(image_files):
        if verbose:
            print(f"Processing image {idx + 1}/{len(image_files)}: {image_path.name}")

        img = cv2.imread(str(image_path))

        if img is None:
            if verbose:
                print(f"Warning: Could not read {image_path}, skipping...")
            continue

        # Resize if necessary
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # Write the same frame multiple times
        for _ in range(frames_per_image):
            out.write(img)

    out.release()

    if verbose:
        print(f"\n✓ Video created successfully: {output_video}")
        print(f"  Total frames: {len(image_files) * frames_per_image}")

    return str(output_video)


def create_video_with_transitions(
    image_folder: Union[str, Path],
    output_video: Union[str, Path],
    fps: int = 30,
    frames_per_image: int = 30,
    transition_frames: int = 15,
    image_pattern: str = "*.png",
    sort_key: Optional[Callable] = None,
    verbose: bool = True,
) -> str:
    """
    Create a video with smooth crossfade transitions between images.

    Parameters
    ----------
    image_folder : str or Path
        Path to folder containing images
    output_video : str or Path
        Path for output video file
    fps : int, default=30
        Frames per second
    frames_per_image : int, default=30
        Number of frames each image persists (excluding transition)
    transition_frames : int, default=15
        Number of frames for crossfade transition between images
    image_pattern : str, default='*.png'
        Glob pattern to match image files
    sort_key : callable, optional
        Function to sort images
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    str
        Path to the created video file

    Examples
    --------
    >>> create_video_with_transitions(
    ...     'my_images/',
    ...     'smooth_video.mp4',
    ...     transition_frames=20
    ... )
    """
    image_folder = Path(image_folder)
    image_files = sorted(image_folder.glob(image_pattern), key=sort_key if sort_key else lambda x: x.name)

    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    if verbose:
        total_frames = len(image_files) * frames_per_image + (len(image_files) - 1) * transition_frames
        duration = total_frames / fps
        print(f"Found {len(image_files)} images")
        print(f"Video settings: {fps} FPS, {frames_per_image} frames per image, {transition_frames} transition frames")
        print(f"Total video duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")

    # Read first image
    first_image = cv2.imread(str(image_files[0]))
    height, width = first_image.shape[:2]

    if verbose:
        print(f"Video dimensions: {width}x{height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    for idx in range(len(image_files)):
        if verbose:
            print(f"Processing image {idx + 1}/{len(image_files)}: {image_files[idx].name}")

        current_img = cv2.imread(str(image_files[idx]))
        if current_img.shape[:2] != (height, width):
            current_img = cv2.resize(current_img, (width, height))

        # Show current image
        for _ in range(frames_per_image):
            out.write(current_img)

        # Add transition to next image if not last
        if idx < len(image_files) - 1:
            next_img = cv2.imread(str(image_files[idx + 1]))
            if next_img.shape[:2] != (height, width):
                next_img = cv2.resize(next_img, (width, height))

            # Crossfade
            for t in range(transition_frames):
                alpha = t / transition_frames
                blended = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
                out.write(blended)

    out.release()

    if verbose:
        print(f"\n✓ Video with transitions created: {output_video}")

    return str(output_video)


def create_video_from_file_list(
    image_files: List[Union[str, Path]],
    output_video: Union[str, Path],
    fps: int = 30,
    frames_per_image: int = 30,
    verbose: bool = True,
) -> str:
    """
    Create a video from an explicit list of image files.
    Useful when you want complete control over image order.

    Parameters
    ----------
    image_files : list of str or Path
        Ordered list of image file paths
    output_video : str or Path
        Path for output video file
    fps : int, default=30
        Frames per second
    frames_per_image : int, default=30
        Number of frames each image persists
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    str
        Path to the created video file

    Examples
    --------
    >>> files = ['frame_000.png', 'frame_001.png', 'frame_002.png']
    >>> create_video_from_file_list(files, 'output.mp4', frames_per_image=45)
    """
    image_files = [Path(f) for f in image_files]

    if not image_files:
        raise ValueError("No images provided")

    if verbose:
        duration = len(image_files) * frames_per_image / fps
        print(f"Creating video from {len(image_files)} images")
        print(f"Total duration: {duration:.2f} seconds")

    # Read first image
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        raise ValueError(f"Could not read image: {image_files[0]}")

    height, width = first_image.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    for idx, image_path in enumerate(image_files):
        if verbose:
            print(f"Processing {idx + 1}/{len(image_files)}: {image_path.name}")

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        for _ in range(frames_per_image):
            out.write(img)

    out.release()

    if verbose:
        print(f"\n✓ Video created: {output_video}")

    return str(output_video)


# Utility sorting functions
def sort_by_number_in_filename(path: Path) -> int:
    """
    Extract number from filename for sorting.
    Works with patterns like: 'image_001.png', 'frame_042.png', 'circular_60.png'
    """
    import re

    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[0]) if numbers else 0


def sort_by_modification_time(path: Path) -> float:
    """Sort by file modification time (oldest first)"""
    return path.stat().st_mtime


# Example usage
if __name__ == "__main__":
    # Example: Create a simple video
    create_video_from_images(
        image_folder="example_images",
        output_video="test_video.mp4",
        fps=30,
        frames_per_image=30,
    )
