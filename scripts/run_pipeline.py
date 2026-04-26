#!/usr/bin/env python3
"""End-to-end visualsync pipeline for a single scene.

Steps (in order):
  0. scripts/download_weights.sh     — fetch SAM2 / DEVA / MASt3R checkpoints
  1. extract_frames.py        — raw_data/<scene>/*.mp4 → data/<scene>/<scene>_camN/rgb/
  2. write gpt_video/tags.json into every cam dir (broadcast same prompt)
  3. preprocess/run_dino_sam2.py     — gsam2 masks per cam
  4. preprocess/vggt_to_colmap.py    — VGGT poses + COLMAP under data/<scene>
  5. analysis/merge_images_to_videos.py  — rgb/mask → data/<scene>/videos/*.mp4
  6. analysis/run_cotracker.py       — per-cam cotracker tracks
  7. analysis/match_tracks_mast3r.py — cross-cam MASt3R correspondences
  8. sync/shaowei_sync_v6.py         — sync offset estimation per cam pair

Use --skip step1,step2,... to skip steps that have already been run.
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

DEFAULT_TAGS = {
    "dynamic": ["man", "woman", "pipette", "hand"],
    "reasoning": "detailed explanation of how you identified movement between frames for each object type",
}

STEPS = [
    "weights",
    "extract",
    "tags",
    "sam2",
    "vggt",
    "merge",
    "cotracker",
    "match",
    "sync",
]


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(map(str, cmd))}\n", flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=cwd or REPO)
    if result.returncode != 0:
        sys.exit(
            f"\n[run_pipeline] step failed (exit {result.returncode}): {' '.join(map(str, cmd))}"
        )


def find_cams(scene_dir: Path):
    return sorted(
        p
        for p in scene_dir.iterdir()
        if p.is_dir() and p.name.startswith(scene_dir.name + "_cam")
    )


def cam_rgb_count(cam: Path) -> int:
    rgb = cam / "rgb"
    if not rgb.exists():
        return 0
    return sum(
        1 for _ in rgb.iterdir() if _.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def assert_rgb_ready(cams, step: str):
    """Bail out early with a clear message if rgb frames are missing."""
    bad = [(cam, cam_rgb_count(cam)) for cam in cams]
    bad = [(c, n) for c, n in bad if n == 0]
    if not bad:
        return
    print(
        f"\n[run_pipeline] cannot start '{step}': missing rgb frames", file=sys.stderr
    )
    for cam, _ in bad:
        rgb = cam / "rgb"
        if rgb.exists():
            print(f"  - {rgb} exists but is empty", file=sys.stderr)
        else:
            print(f"  - {rgb} does not exist", file=sys.stderr)
    print(
        "\nRe-run the 'extract' step (drop it from --skip), or copy the frames "
        "into <scene>/<scene>_camN/rgb/ manually.",
        file=sys.stderr,
    )
    sys.exit(2)


def write_tags(cam_dirs, tags):
    for cam in cam_dirs:
        gpt = cam / "gpt_video"
        gpt.mkdir(parents=True, exist_ok=True)
        out = gpt / "tags.json"
        out.write_text(json.dumps(tags, indent=2))
        print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="e.g. scene1")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument(
        "--tags-file",
        type=Path,
        default=None,
        help="JSON file with {dynamic: [...]} broadcast to every cam (default: built-in)",
    )
    parser.add_argument("--cotracker-grid-size", type=int, default=100)
    parser.add_argument(
        "--cotracker-model-type", choices=["online", "offline"], default="offline"
    )
    parser.add_argument(
        "--cotracker-device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device for run_cotracker.py (default: auto = cuda > mps > cpu).",
    )
    parser.add_argument(
        "--cotracker-cpu-fallback",
        dest="cotracker_cpu_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If CUDA OOMs, retry that instance on CPU. Default: enabled. "
        "Pass --no-cotracker-cpu-fallback to abort instead.",
    )
    parser.add_argument(
        "--vggt-suffix",
        default="300",
        help="frame count suffix for camera_parameters_<suffix>.npz "
        "(consumed by both match_tracks_mast3r and shaowei_sync_v6)",
    )
    parser.add_argument("--offset-range", type=int, default=150)
    parser.add_argument(
        "--pairs",
        default=None,
        help="comma-separated cam-index pairs for match/sync, "
        "e.g. '1-2,1-3,2-3' (default: all unordered pairs)",
    )
    parser.add_argument(
        "--skip",
        default="",
        help=f"comma-separated steps to skip; one or more of: {', '.join(STEPS)}",
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated steps to run (skip everything else)",
    )
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    bad = (skip | only) - set(STEPS)
    if bad:
        sys.exit(f"unknown step(s): {sorted(bad)}; valid: {STEPS}")

    def should_run(step):
        if only:
            return step in only
        return step not in skip

    scene = args.scene
    scene_dir = args.data_dir / scene
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Download model weights ────────────────────────────────────────────
    if should_run("weights"):
        run(["bash", REPO / "scripts/download_weights.sh"])

    # ── 1. Extract frames ────────────────────────────────────────────────────
    if should_run("extract"):
        run(
            [
                PY,
                REPO / "data/extract_frames.py",
                "--data-dir",
                args.data_dir,
                "--scene",
                scene,
            ]
        )

    if not scene_dir.exists():
        sys.exit(f"scene dir not found after extract: {scene_dir}")

    cams = find_cams(scene_dir)
    if not cams:
        sys.exit(f"no cam dirs (expected {scene}_cam1, ...) under {scene_dir}")
    print(
        f"\n[run_pipeline] cams: "
        + ", ".join(f"{c.name} ({cam_rgb_count(c)} frames)" for c in cams)
    )

    # ── 2. Write tags.json ───────────────────────────────────────────────────
    if should_run("tags"):
        if args.tags_file:
            tags = json.loads(args.tags_file.read_text())
        else:
            tags = DEFAULT_TAGS
        print("\n[run_pipeline] writing gpt_video/tags.json into each cam")
        write_tags(cams, tags)

    # ── 3. Grounded-SAM2 ─────────────────────────────────────────────────────
    if should_run("sam2"):
        run([PY, REPO / "preprocess/run_dino_sam2.py", "--workdir", scene_dir])

    # ── 4. VGGT + COLMAP (vis_path defaults to workdir now) ──────────────────
    if should_run("vggt"):
        assert_rgb_ready(cams, "vggt")
        run(
            [
                PY,
                REPO / "preprocess/vggt_to_colmap.py",
                "--workdir",
                scene_dir,
                "--save_colmap",
            ]
        )

    # ── 5. Merge frames into mp4 (cotracker input) ───────────────────────────
    if should_run("merge"):
        assert_rgb_ready(cams, "merge")
        run(
            [
                PY,
                REPO / "analysis/merge_images_to_videos.py",
                "--scene",
                scene,
                "--data-dir",
                args.data_dir,
            ]
        )

    # ── 6. CoTracker per camera ──────────────────────────────────────────────
    if should_run("cotracker"):
        for cam in cams:
            video_path = scene_dir / "videos" / f"{cam.name}_rgb.mp4"
            mask_dir = cam / "gsam2" / "mask"
            if not video_path.exists():
                print(f"[skip cotracker {cam.name}] missing {video_path}")
                continue
            if not mask_dir.exists():
                print(f"[skip cotracker {cam.name}] missing {mask_dir}")
                continue
            cot_cmd = [
                PY,
                REPO / "analysis/run_cotracker.py",
                "--video_path",
                video_path,
                "--mask_dir",
                mask_dir,
                "--grid_size",
                args.cotracker_grid_size,
                "--model-type",
                args.cotracker_model_type,
                "--device",
                args.cotracker_device,
            ]
            cot_cmd.append(
                "--cpu-fallback" if args.cotracker_cpu_fallback else "--no-cpu-fallback"
            )
            run(cot_cmd)

    # ── Build the cam-pair list ──────────────────────────────────────────────
    cam_indices = []
    for c in cams:
        suffix = c.name[len(scene) + len("_cam") :]
        if suffix.isdigit():
            cam_indices.append(int(suffix))
    cam_indices.sort()

    if args.pairs:
        pairs = []
        for token in args.pairs.split(","):
            a, b = token.strip().split("-")
            pairs.append((int(a), int(b)))
    else:
        pairs = list(itertools.combinations(cam_indices, 2))

    pair_names = [(f"{scene}_cam{a}", f"{scene}_cam{b}") for a, b in pairs]
    print(f"\n[run_pipeline] cam pairs: {pair_names}")

    # ── 7. MASt3R cross-cam matching ─────────────────────────────────────────
    if should_run("match"):
        for cam1, cam2 in pair_names:
            run(
                [
                    PY,
                    REPO / "analysis/match_tracks_mast3r.py",
                    "--dataset_root",
                    scene_dir,
                    "--result_root",
                    results_dir,
                    "--cam1_name",
                    cam1,
                    "--cam2_name",
                    cam2,
                    "--vggt_suffix",
                    args.vggt_suffix,
                    "--visualize",
                ]
            )

    # ── 8. Sync (offset estimation) ──────────────────────────────────────────
    if should_run("sync"):
        for cam1, cam2 in pair_names:
            run(
                [
                    PY,
                    REPO / "sync/shaowei_sync_v6.py",
                    "--dataset_root",
                    scene_dir,
                    "--result_root",
                    results_dir,
                    "--video1_name",
                    cam1,
                    "--video2_name",
                    cam2,
                    "--use_vggt",
                    "--vggt_choice",
                    args.vggt_suffix,
                    "--use_v2",
                    "--disable_gt",
                    "--offset_range",
                    args.offset_range,
                ]
            )

    print("\n[run_pipeline] done.")


if __name__ == "__main__":
    main()
