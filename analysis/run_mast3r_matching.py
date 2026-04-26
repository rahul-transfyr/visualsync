#!/usr/bin/env python
"""
Run MASt3R to establish cross-view correspondences between camera pairs.

This script:
1. Loads sampled keyframes from two videos
2. Uses MASt3R to find 2D-2D matches between views
3. Links these matches to CoTracker tracklets
4. Saves spatio-temporal correspondences for synchronization
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

# Add mast3r and its dust3r submodule to path
_REPO = os.path.join(os.path.dirname(__file__), "..", "mast3r")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "dust3r"))

from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R


def load_frame(frame_path, size=512):
    """Load a single frame."""
    img = Image.open(frame_path)
    # Resize while maintaining aspect ratio
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img)


def sample_keyframes(video_dir, num_samples=10):
    """Sample evenly-spaced keyframes from video."""
    frame_paths = sorted(glob.glob(os.path.join(video_dir, "rgb", "*.jpg")))

    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {video_dir}/rgb/")

    # Sample evenly
    indices = np.linspace(0, len(frame_paths) - 1, num_samples, dtype=int)
    sampled_paths = [frame_paths[i] for i in indices]
    sampled_indices = indices.tolist()

    return sampled_paths, sampled_indices


def match_frames(model, frame1_path, frame2_path, device, size=512):
    """Match two frames using MASt3R."""
    images = load_images([frame1_path, frame2_path], size=size)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]

    desc1 = pred1["desc"].squeeze(0).detach()
    desc2 = pred2["desc"].squeeze(0).detach()

    # Find 2D-2D matches
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1,
        desc2,
        subsample_or_initxy1=8,
        device=device,
        dist="dot",
        block_size=2**13,
    )

    # Filter matches near borders
    H0, W0 = view1["true_shape"][0]
    H1, W1 = view2["true_shape"][0]

    valid_matches_im0 = (
        (matches_im0[:, 0] >= 3)
        & (matches_im0[:, 0] < int(W0) - 3)
        & (matches_im0[:, 1] >= 3)
        & (matches_im0[:, 1] < int(H0) - 3)
    )

    valid_matches_im1 = (
        (matches_im1[:, 0] >= 3)
        & (matches_im1[:, 0] < int(W1) - 3)
        & (matches_im1[:, 1] >= 3)
        & (matches_im1[:, 1] < int(H1) - 3)
    )

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]

    # Convert to numpy if torch tensor
    if torch.is_tensor(matches_im0):
        matches_im0 = matches_im0.cpu().numpy()
    if torch.is_tensor(matches_im1):
        matches_im1 = matches_im1.cpu().numpy()

    return matches_im0, matches_im1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam1_dir", required=True, help="Directory for camera 1")
    parser.add_argument("--cam2_dir", required=True, help="Directory for camera 2")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save correspondences (default: cam1_dir/mast3r)",
    )
    parser.add_argument(
        "--num_keyframes",
        type=int,
        default=10,
        help="Number of keyframes to sample (default: 10)",
    )
    parser.add_argument(
        "--model_name",
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        help="MASt3R model name",
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="Image size for processing"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.cam1_dir, "mast3r")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading MASt3R model: {args.model_name}")
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(device)
    print(f"Using device: {device}")

    # Sample keyframes
    print(f"\nSampling {args.num_keyframes} keyframes from each camera...")
    cam1_frames, cam1_indices = sample_keyframes(args.cam1_dir, args.num_keyframes)
    cam2_frames, cam2_indices = sample_keyframes(args.cam2_dir, args.num_keyframes)

    print(f"Camera 1: {len(cam1_frames)} frames from {args.cam1_dir}")
    print(f"Camera 2: {len(cam2_frames)} frames from {args.cam2_dir}")

    # Match all keyframe pairs
    all_correspondences = []

    for i, (frame1_path, frame1_idx) in enumerate(zip(cam1_frames, cam1_indices)):
        for j, (frame2_path, frame2_idx) in enumerate(zip(cam2_frames, cam2_indices)):
            print(f"\nMatching cam1[{frame1_idx}] <-> cam2[{frame2_idx}]...", end=" ")

            try:
                matches1, matches2 = match_frames(
                    model, frame1_path, frame2_path, device, args.image_size
                )

                num_matches = len(matches1)
                print(f"✓ {num_matches} matches")

                if num_matches > 0:
                    all_correspondences.append(
                        {
                            "cam1_frame_idx": frame1_idx,
                            "cam2_frame_idx": frame2_idx,
                            "cam1_points": matches1,  # Nx2 array
                            "cam2_points": matches2,  # Nx2 array
                            "num_matches": num_matches,
                        }
                    )

            except Exception as e:
                print(f"✗ Failed: {e}")
                continue

    # Save correspondences
    output_file = os.path.join(output_dir, "cross_view_correspondences.npz")

    # Convert to arrays for saving
    np.savez(
        output_file,
        correspondences=all_correspondences,
        cam1_dir=args.cam1_dir,
        cam2_dir=args.cam2_dir,
        num_keyframes=args.num_keyframes,
    )

    print(
        f"\n✓ Saved {len(all_correspondences)} keyframe pair matches to {output_file}"
    )
    print(f"  Total matches: {sum(c['num_matches'] for c in all_correspondences)}")
