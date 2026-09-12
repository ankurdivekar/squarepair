"""Combine a directory of solution images into a video with crossfade transitions."""

from pathlib import Path
from typing import Callable, Optional, Union

import cv2


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
    """Create a video with crossfade transitions between images in ``image_folder``.

    Each image holds for ``frames_per_image`` frames, then crossfades into the next
    over ``transition_frames`` frames. Images are sorted alphabetically by filename
    unless ``sort_key`` is given.
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

    first_image = cv2.imread(str(image_files[0]))
    height, width = first_image.shape[:2]

    if verbose:
        print(f"Video dimensions: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    for idx in range(len(image_files)):
        if verbose:
            print(f"Processing image {idx + 1}/{len(image_files)}: {image_files[idx].name}")

        current_img = cv2.imread(str(image_files[idx]))
        if current_img.shape[:2] != (height, width):
            current_img = cv2.resize(current_img, (width, height))

        for _ in range(frames_per_image):
            out.write(current_img)

        if idx < len(image_files) - 1:
            next_img = cv2.imread(str(image_files[idx + 1]))
            if next_img.shape[:2] != (height, width):
                next_img = cv2.resize(next_img, (width, height))

            for t in range(transition_frames):
                alpha = t / transition_frames
                blended = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
                out.write(blended)

    out.release()

    if verbose:
        print(f"\n✓ Video with transitions created: {output_video}")

    return str(output_video)
