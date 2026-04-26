#!/usr/bin/env python
# Run CoTracker3 on videos with per-frame masks for VisualSync preprocessing

import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "co-tracker"))

from cotracker.utils.visualizer import Visualizer, read_video_from_path


def pick_device(name: str) -> str:
    """Resolve a --device choice, falling back if the requested device isn't available."""
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("Requested cuda but no CUDA available; falling back to cpu")
        return "cpu"
    return name


def is_oom(err: Exception) -> bool:
    """Detect a CUDA OOM regardless of torch version (some versions only raise RuntimeError)."""
    if hasattr(torch, "cuda") and isinstance(
        err, getattr(torch.cuda, "OutOfMemoryError", ())
    ):
        return True
    msg = str(err).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def load_masks_from_dir(mask_dir, num_frames):
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if len(mask_paths) == 0:
        print(f"Warning: No mask files found in {mask_dir}")
        return None

    if len(mask_paths) != num_frames:
        print(f"Warning: Found {len(mask_paths)} masks but {num_frames} video frames")

    masks = []
    for path in mask_paths[:num_frames]:
        mask = np.array(Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        masks.append(mask)

    return np.stack(masks)  # (T, H, W)


def get_unique_instances(masks):
    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids > 0]
    return unique_ids.tolist()


def run_offline(model, video_cpu, segm_mask_cpu, args, device):
    """Run offline CoTracker — full video on device at once."""
    video = video_cpu.to(device)
    segm_mask = segm_mask_cpu.to(device) if segm_mask_cpu is not None else None

    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            video,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            backward_tracking=args.backward_tracking,
            segm_mask=segm_mask,
        )
    return pred_tracks, pred_visibility


def run_online(model, video_cpu, segm_mask_cpu, args, device):
    """Run online CoTracker — processes one window at a time to save memory."""
    num_frames = video_cpu.shape[1]
    is_first_step = True
    pred_tracks = pred_visibility = None

    for ind in range(0, num_frames - model.step, model.step):
        chunk = video_cpu[:, ind : ind + model.step * 2].to(device)
        segm_chunk = None
        if segm_mask_cpu is not None:
            segm_chunk = segm_mask_cpu[:, ind : ind + model.step * 2].to(device)

        with torch.no_grad():
            pred_tracks, pred_visibility = model(
                chunk,
                is_first_step=is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
        is_first_step = False

        # Free chunk from device immediately
        del chunk
        if segm_chunk is not None:
            del segm_chunk
        if device == "cuda":
            torch.cuda.empty_cache()

    return pred_tracks, pred_visibility


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="path to video file")
    parser.add_argument(
        "--mask_dir",
        default=None,
        help="path to directory containing per-frame mask images",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory to save tracking results (default: same as video directory)",
    )
    parser.add_argument(
        "--grid_size", type=int, default=50, help="Regular grid size (default: 50)"
    )
    parser.add_argument(
        "--grid_query_frame", type=int, default=0, help="Frame to sample points from"
    )
    parser.add_argument(
        "--backward_tracking", action="store_true", help="Track in both directions"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save visualization videos"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)",
    )
    parser.add_argument(
        "--model-type",
        choices=["online", "offline"],
        default="online",
        help="online uses a sliding window (less memory); offline loads the full video (default: online)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. 'auto' picks cuda > cpu (default: auto).",
    )
    parser.add_argument(
        "--cpu-fallback",
        dest="cpu_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a CUDA OOM hits, retry that instance on CPU. Default: enabled. "
        "Pass --no-cpu-fallback to abort instead.",
    )

    args = parser.parse_args()
    device = pick_device(args.device)

    # Load video (kept as uint8 on CPU until needed)
    print(f"Loading video from {args.video_path}...")
    video_np = read_video_from_path(args.video_path)
    if args.max_frames is not None:
        video_np = video_np[: args.max_frames]
        print(f"  Limiting to {args.max_frames} frames")

    num_frames = video_np.shape[0]
    # Keep on CPU as float; move to device per-chunk in online mode
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()  # B T C H W
    print(f"Loaded {num_frames} frames, shape: {video.shape}")
    del video_np

    # Load masks
    masks = None
    instance_ids = [None]
    if args.mask_dir and os.path.isdir(args.mask_dir):
        print(f"Loading masks from {args.mask_dir}...")
        masks = load_masks_from_dir(args.mask_dir, num_frames)
        if masks is not None:
            instance_ids = get_unique_instances(masks)
            print(f"Found {len(instance_ids)} dynamic object instances: {instance_ids}")

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Expected layout:
        #   <dataset_root>/videos/<cam>_rgb.mp4  →  <dataset_root>/<cam>/cotracker
        # Falls back to <video_dir>/<scene>/cotracker if the layout doesn't match.
        video_dir = os.path.dirname(args.video_path)
        scene_name = os.path.basename(args.video_path).replace(".mp4", "")
        cam_name = (
            scene_name[: -len("_rgb")] if scene_name.endswith("_rgb") else scene_name
        )
        if os.path.basename(video_dir) == "videos":
            dataset_root = os.path.dirname(video_dir)
            output_dir = os.path.join(dataset_root, cam_name, "cotracker")
        else:
            output_dir = os.path.join(video_dir, cam_name, "cotracker")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"Loading CoTracker3 ({args.model_type}) model...")
    model_name = (
        "cotracker3_online" if args.model_type == "online" else "cotracker3_offline"
    )
    model = torch.hub.load("facebookresearch/co-tracker", model_name)
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    cpu_model_holder = [None]  # list cell so the closure can mutate it

    def get_cpu_model():
        if cpu_model_holder[0] is None:
            print("  Loading CPU copy of model for OOM fallback...")
            m = torch.hub.load("facebookresearch/co-tracker", model_name)
            m = m.to("cpu")
            m.eval()
            cpu_model_holder[0] = m
        return cpu_model_holder[0]

    run_fn = run_online if args.model_type == "online" else run_offline

    for instance_id in instance_ids:
        if instance_id is None:
            print(f"\nTracking without mask (grid_size={args.grid_size})...")
            segm_mask_cpu = None
            suffix = ""
            effective_query_frame = args.grid_query_frame
        else:
            print(f"\nTracking instance {instance_id} (grid_size={args.grid_size})...")
            instance_mask = (masks == instance_id).astype(np.uint8)
            suffix = f"_instance{instance_id}"
            # Find first frame where this instance is visible
            first_visible = next(
                (i for i in range(instance_mask.shape[0]) if instance_mask[i].any()),
                args.grid_query_frame,
            )
            effective_query_frame = first_visible
            if first_visible != args.grid_query_frame:
                print(
                    f"  Instance not visible at frame {args.grid_query_frame}; using frame {first_visible} as query frame"
                )
            # CoTracker expects (B, 1, H, W) — a single spatial mask for the query frame
            segm_mask_cpu = torch.from_numpy(instance_mask[effective_query_frame])[
                None, None
            ].float()

        # Override grid_query_frame for this instance
        original_query_frame = args.grid_query_frame
        args.grid_query_frame = effective_query_frame

        active_device = device
        try:
            pred_tracks, pred_visibility = run_fn(
                model, video, segm_mask_cpu, args, active_device
            )
        except Exception as e:
            if not (active_device == "cuda" and is_oom(e)):
                raise
            torch.cuda.empty_cache()
            # The offline model loads the full clip onto the GPU at once and
            # is the variant that reliably OOMs, so we always retry on CPU
            # there. Online OOMs still require an explicit --cpu-fallback
            # opt-in (something else is wrong if a per-window forward OOMs).
            offline_auto_fallback = args.model_type == "offline"
            if not (offline_auto_fallback or args.cpu_fallback):
                args.grid_query_frame = original_query_frame
                msg = (
                    f"  CUDA OOM on instance {instance_id} (model-type=online). "
                    f"Re-run with --cpu-fallback or a smaller --grid_size."
                )
                raise RuntimeError(msg) from e
            reason = (
                "offline model — auto-falling back to CPU"
                if offline_auto_fallback
                else "falling back to CPU (--cpu-fallback)"
            )
            print(f"  CUDA OOM on instance {instance_id}; {reason}")
            active_device = "cpu"
            pred_tracks, pred_visibility = run_fn(
                get_cpu_model(), video, segm_mask_cpu, args, active_device
            )

        args.grid_query_frame = original_query_frame

        if pred_tracks is None or pred_tracks.shape[2] == 0:
            print(f"  Skipping instance {instance_id} — no tracks found")
            continue

        print(f"  Tracks: {pred_tracks.shape}, visibility: {pred_visibility.shape}")

        output_file = os.path.join(output_dir, f"tracks{suffix}.npz")
        np.savez(
            output_file,
            tracks=pred_tracks.cpu().numpy(),
            visibility=pred_visibility.cpu().numpy(),
            grid_size=args.grid_size,
            grid_query_frame=effective_query_frame,
        )
        print(f"  Saved to {output_file}")

        if args.visualize:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            vis = Visualizer(save_dir=vis_dir, pad_value=120, linewidth=3)
            vis.visualize(
                video.to(active_device),
                pred_tracks,
                pred_visibility,
                query_frame=0 if args.backward_tracking else effective_query_frame,
                filename=f"tracks{suffix}",
            )
            print(f"  Saved visualization to {vis_dir}")

        del pred_tracks, pred_visibility
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nAll tracking complete! Results saved to {output_dir}")
