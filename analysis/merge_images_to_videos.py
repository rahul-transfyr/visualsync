#!/usr/bin/env python3
"""
Create videos from rgb and gsam2/mask image sequences for each camera in a scene.

Usage:
    python merge_images_to_videos.py --scene scene1 [--data-dir /path/to/data] [--fps 30]

Output videos are saved to data/<scene>/videos/:
    <scene>_cam1_rgb.mp4
    <scene>_cam1_mask.mp4
    <scene>_cam2_rgb.mp4
    ...
"""

import argparse
import sys
from pathlib import Path

import cv2


def get_image_files(folder: Path, ext: str) -> list[Path]:
    return sorted(folder.glob(f"*.{ext}"))


def create_video(image_files: list[Path], output_path: Path, fps: int) -> bool:
    if not image_files:
        return False

    first = cv2.imread(str(image_files[0]))
    if first is None:
        print(f"  Could not read {image_files[0]}")
        return False

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"  Could not open video writer for {output_path}")
        return False

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Warning: skipping unreadable frame {img_path.name}")
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Create videos from rgb and mask image sequences.")
    parser.add_argument("--scene", required=True, help="Scene name, e.g. scene1")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Path to the data directory (default: ../data relative to this script)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames per video (default: all)")
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    scene_dir = data_dir / args.scene

    if not scene_dir.exists():
        print(f"ERROR: scene directory not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = scene_dir / "videos"
    output_dir.mkdir(exist_ok=True)

    sources = [
        ("rgb", "jpg"),
        ("gsam2/mask", "png"),
    ]

    any_created = False
    for cam_index in (1, 2, 3):
        cam_name = f"{args.scene}_cam{cam_index}"
        cam_dir = scene_dir / cam_name

        if not cam_dir.exists():
            print(f"Skipping {cam_name} — directory not found")
            continue

        for subpath, ext in sources:
            folder = cam_dir / subpath
            if not folder.exists():
                print(f"Skipping {cam_name}/{subpath} — folder not found")
                continue

            images = get_image_files(folder, ext)
            if not images:
                print(f"Skipping {cam_name}/{subpath} — no .{ext} files found")
                continue

            if args.max_frames is not None:
                images = images[: args.max_frames]

            label = subpath.replace("/", "_")
            output_path = output_dir / f"{cam_name}_{label}.mp4"
            print(f"Processing {cam_name}/{subpath} ({len(images)} frames) -> {output_path.name}")

            if create_video(images, output_path, args.fps):
                print(f"  Created {output_path}")
                any_created = True
            else:
                print(f"  Failed to create video for {cam_name}/{subpath}")

    print()
    if any_created:
        print(f"Videos saved to: {output_dir}")
    else:
        print("No videos created.")


if __name__ == "__main__":
    main()


