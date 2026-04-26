#!/usr/bin/env python3
"""
Extract frames from cam1/cam2/cam3 videos into their respective
<scene>_camX/rgb/ folders.

Usage:
    python extract_frames.py [--data-dir /path/to/data] [--scene scene1] [--ext jpg] [--fps 0]

    --data-dir  Path to the data directory (default: directory of this script)
    --scene     Scene folder prefix (default: scene1)
    --ext       Output image extension: jpg or png (default: jpg)
    --fps       Frames per second to extract; 0 means every frame (default: 0)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def find_video_for_cam(data_dir: Path, cam_index: int) -> Path | None:
    """Return the first video file whose name ends with _camN (before extension)."""
    pattern = re.compile(rf".*_cam{cam_index}\.mp4$", re.IGNORECASE)
    for f in sorted(data_dir.glob("*.mp4")):
        if pattern.match(f.name):
            return f
    return None


def assign_videos(raw_dir: Path, cam_indices=(1, 2, 3)) -> dict[int, Path]:
    """Assign one mp4 per cam.

    Strategy:
      1. For each cam_index, try the `*_cam{N}.mp4` filename convention.
      2. Any cam slots still unfilled get the remaining mp4s (sorted), as a
         fallback for raw_data dirs that don't follow the naming convention.
    """
    all_mp4s = sorted(raw_dir.glob("*.mp4"))
    assigned: dict[int, Path] = {}
    used: set[Path] = set()

    for cam_index in cam_indices:
        match = find_video_for_cam(raw_dir, cam_index)
        if match is not None and match not in used:
            assigned[cam_index] = match
            used.add(match)

    leftovers = [f for f in all_mp4s if f not in used]
    for cam_index in cam_indices:
        if cam_index in assigned:
            continue
        if not leftovers:
            break
        fallback = leftovers.pop(0)
        print(
            f"NOTE: no *_cam{cam_index}.mp4 in {raw_dir}; "
            f"falling back to {fallback.name}"
        )
        assigned[cam_index] = fallback

    return assigned


def extract_frames(
    video_path: Path,
    output_dir: Path,
    ext: str = "jpg",
    fps: float = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / f"%06d.{ext}")

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    if fps > 0:
        cmd += ["-vf", f"fps={fps}"]

    if ext.lower() == "jpg":
        cmd += ["-q:v", "2"]  # high quality JPEG (1=best, 31=worst)

    cmd.append(output_pattern)

    print(f"\n[cam{video_path.stem[-1]}] {video_path.name}")
    print(f"  -> {output_dir}")
    print(f"  Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: ffmpeg failed for {video_path}", file=sys.stderr)
        sys.exit(result.returncode)

    extracted = sorted(output_dir.glob(f"*.{ext}"))
    print(f"  Extracted {len(extracted)} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract video frames into rgb folders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Path to the data directory (default: directory of this script)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene1",
        help="Scene folder prefix (default: scene1)",
    )
    parser.add_argument(
        "--ext",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0,
        help="FPS to extract; 0 = every frame (default: 0)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    raw_dir = data_dir / "raw_data" / args.scene

    if not raw_dir.exists():
        print(f"ERROR: raw data directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    n_videos = len(sorted(raw_dir.glob("*.mp4")))
    if n_videos == 0:
        print(f"ERROR: no .mp4 files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    cam_indices = tuple(range(1, n_videos + 1))
    assigned = assign_videos(raw_dir, cam_indices)
    print(f"Found {n_videos} video(s) in {raw_dir}")

    for cam_index in cam_indices:
        video = assigned.get(cam_index)
        if video is None:
            print(f"WARNING: no video found for cam{cam_index} in {raw_dir}, skipping.")
            continue

        output_dir = data_dir / args.scene / f"{args.scene}_cam{cam_index}" / "rgb"
        extract_frames(video, output_dir, ext=args.ext, fps=args.fps)

    print("\nDone.")


if __name__ == "__main__":
    main()



