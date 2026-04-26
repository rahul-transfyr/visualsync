#!/usr/bin/env python3
"""
Matches cotracker grid points across cameras using MASt3R feature descriptors.

For each cam2 instance, runs MASt3R on:
  (cam1 query frame, cam2 instance query frame)

Extracts dense descriptors at the exact cotracker pixel positions, then finds
reciprocal nearest-neighbour matches in feature space. This gives direct
cotracker-to-cotracker index pairs (filtered_corr_indices) without any
pixel-snapping approximation.

Also subsets VGGT camera parameters to the cotracker track length.

Outputs:
  {result_root}/scene1/scene1_cam1__scene1_cam2/tracks_match_v2.npz
  {dataset_root}/scene1_cam{N}/vggt/camera_parameters_{vggt_suffix}.npz
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_REPO = os.path.join(os.path.dirname(__file__), "..", "mast3r")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "dust3r"))
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R

# ── helpers ──────────────────────────────────────────────────────────────────


def load_cotracker(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    tracks = d["tracks"][0]  # (T, N, 2)  squeeze batch dim
    visibility = d["visibility"][0]  # (T, N)
    query_frame = int(d["grid_query_frame"])
    return tracks, visibility, query_frame


def discover_instances(cotracker_dir):
    """Return sorted list of instance ids from tracks_instance<id>.npz files."""
    paths = sorted(glob.glob(os.path.join(cotracker_dir, "tracks_instance*.npz")))
    insts = []
    for p in paths:
        name = os.path.basename(p)
        inst = name[len("tracks_instance") : -len(".npz")]
        if inst:
            insts.append(inst)
    return insts


def get_instance_labels(cam_dir):
    """Map cotracker pixel-id -> GSAM2 label using gsam2/mask/*.json metadata.

    The mask PNGs use (json_id % 256) as the per-pixel id, so we collapse
    json ids by mod 256 to recover the (small) ids that name tracks_instance*.npz.
    """
    json_paths = sorted(glob.glob(os.path.join(cam_dir, "gsam2/mask/*.json")))
    px_to_label = {}
    for p in json_paths:
        try:
            for o in json.load(open(p)):
                px_to_label[int(o["id"]) % 256] = o["label"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return px_to_label


def load_instances(cotracker_dir, instance_ids, label_map, cam_label):
    """Load and concatenate cotracker outputs for the given instance ids."""
    parts, vis_parts, qfs, offsets, kept, labels = [], [], [], [], [], []
    offset = 0
    for inst in instance_ids:
        p = os.path.join(cotracker_dir, f"tracks_instance{inst}.npz")
        if not os.path.exists(p):
            print(f"  Warning: {p} not found, skipping")
            continue
        t, v, qf = load_cotracker(p)
        parts.append(t)
        vis_parts.append(v)
        qfs.append(qf)
        offsets.append(offset)
        offset += t.shape[1]
        kept.append(inst)
        try:
            label = label_map.get(int(inst))
        except (TypeError, ValueError):
            label = None
        labels.append(label)
        print(f"{cam_label} instance{inst} ({label}): {t.shape}, query_frame={qf}")
    if not parts:
        return None
    tracks = np.concatenate(parts, axis=1)  # (T, N_total, 2)
    vis = np.concatenate(vis_parts, axis=1)  # (T, N_total)
    return {
        "tracks": tracks,
        "vis": vis,
        "parts": parts,
        "qfs": qfs,
        "offsets": offsets,
        "instances": kept,
        "labels": labels,
    }


def labels_compatible(a, b):
    """Loose label compatibility — exact match or one being a prefix of the other.
    Handles GSAM2 quirks like 'pipe' vs 'pipette'."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    return a.startswith(b) or b.startswith(a)


def frame_path(cam_dir, frame_idx):
    """Return the jpg path for a zero-based frame index."""
    paths = sorted(glob.glob(os.path.join(cam_dir, "rgb", "*.jpg")))
    return paths[frame_idx]


def get_image_size(cam_dir):
    paths = sorted(glob.glob(os.path.join(cam_dir, "rgb", "*.jpg")))
    w, h = Image.open(paths[0]).size
    return w, h  # (W, H)


def mast3r_scale(orig_w, orig_h, mast3r_size=512):
    """Scale factor from original image space to MASt3R resized space."""
    scale = mast3r_size / max(orig_w, orig_h)
    return scale


def extract_features_at_points(desc, pts_xy, orig_w, orig_h, mast3r_size=512):
    """
    Sample MASt3R descriptor map at given pixel coordinates.

    desc   : torch.Tensor (H_m, W_m, D)  — MASt3R feature map (single image)
    pts_xy : np.ndarray  (N, 2)          — (x, y) in original image pixels
    Returns: torch.Tensor (N, D)
    """
    H_m, W_m, D = desc.shape
    scale = mast3r_size / max(orig_w, orig_h)
    x = (pts_xy[:, 0] * scale).round().clip(0, W_m - 1).astype(int)
    y = (pts_xy[:, 1] * scale).round().clip(0, H_m - 1).astype(int)
    return desc[y, x]  # (N, D)


def reciprocal_nn(feat1, feat2, min_score=0.0):
    """
    Reciprocal nearest-neighbour matching on L2-normalised descriptors.

    feat1, feat2 : torch.Tensor (N1, D), (N2, D)
    Returns (idx1, idx2, scores) : matched indices and their cosine similarities
    """
    f1 = F.normalize(feat1.float(), dim=-1)
    f2 = F.normalize(feat2.float(), dim=-1)
    sim = f1 @ f2.T  # (N1, N2)
    nn12 = sim.argmax(dim=1)  # (N1,) best cam2 match for each cam1 point
    nn21 = sim.argmax(dim=0)  # (N2,) best cam1 match for each cam2 point

    idx1 = torch.arange(len(feat1), device=sim.device)
    reciprocal = nn21[nn12] == idx1

    if min_score > 0.0:
        scores_all = sim[idx1, nn12]
        reciprocal &= scores_all >= min_score

    kept_i1 = idx1[reciprocal]
    kept_i2 = nn12[reciprocal]
    kept_scores = sim[kept_i1, kept_i2]
    return (
        kept_i1.cpu().numpy(),
        kept_i2.cpu().numpy(),
        kept_scores.cpu().numpy(),
    )


def make_obj_array(d):
    arr = np.empty((), dtype=object)
    arr[()] = d
    return arr


def visualize_matches(
    img1_path, img2_path, pts1, pts2, idx1, idx2, out_path, max_lines=200
):
    """
    Save a side-by-side image with lines connecting matched cotracker points.
    pts1, pts2 : (N, 2) arrays of (x, y) in original image pixels
    idx1, idx2 : matched index arrays into pts1 and pts2
    """
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    W1, H1 = img1.size
    W2, H2 = img2.size
    H = max(H1, H2)
    canvas = Image.new("RGB", (W1 + W2, H))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (W1, 0))
    draw = ImageDraw.Draw(canvas)

    # Sample a subset so the image isn't overcrowded
    n = min(len(idx1), max_lines)
    rng = np.random.default_rng(0)
    sel = (
        rng.choice(len(idx1), size=n, replace=False)
        if len(idx1) > n
        else np.arange(len(idx1))
    )

    for s in sel:
        x1, y1 = pts1[idx1[s]]
        x2, y2 = pts2[idx2[s]]
        color = tuple(rng.integers(60, 255, size=3).tolist())
        draw.line([(x1, y1), (W1 + x2, y2)], fill=color, width=1)
        draw.ellipse([(x1 - 3, y1 - 3), (x1 + 3, y1 + 3)], fill=color)
        draw.ellipse([(W1 + x2 - 3, y2 - 3), (W1 + x2 + 3, y2 + 3)], fill=color)

    canvas.save(out_path)
    print(f"  Saved viz: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", default="/Users/rsvarma/Code/visualsync/data/scene1"
    )
    parser.add_argument(
        "--result_root", default="/Users/rsvarma/Code/visualsync/results"
    )
    parser.add_argument("--cam1_name", default="scene1_cam1")
    parser.add_argument("--cam2_name", default="scene1_cam2")
    parser.add_argument(
        "--cam1_instances",
        default="",
        help="Comma-separated GSAM2 instance ids for cam1 (empty = auto-discover)",
    )
    parser.add_argument(
        "--cam2_instances",
        default="",
        help="Comma-separated GSAM2 instance ids for cam2 (empty = auto-discover)",
    )
    parser.add_argument(
        "--model_name", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Min cosine similarity to keep a single reciprocal match (0 = keep all reciprocal)",
    )
    parser.add_argument(
        "--min_matches",
        type=int,
        default=20,
        help="Drop instance pairs with fewer than this many reciprocal matches",
    )
    parser.add_argument(
        "--min_pair_mean_score",
        type=float,
        default=0.5,
        help="Drop instance pairs whose mean cosine similarity over kept matches is below this",
    )
    parser.add_argument(
        "--ignore_labels",
        action="store_true",
        help="Skip GSAM2 label compatibility check (try every cam1 x cam2 instance pair)",
    )
    parser.add_argument("--vggt_suffix", default="300")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save side-by-side match visualization images",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    cam1_dir = os.path.join(args.dataset_root, args.cam1_name)
    cam2_dir = os.path.join(args.dataset_root, args.cam2_name)
    orig_w1, orig_h1 = get_image_size(cam1_dir)
    orig_w2, orig_h2 = get_image_size(cam2_dir)
    print(f"cam1 image size: {orig_w1}x{orig_h1}")
    print(f"cam2 image size: {orig_w2}x{orig_h2}")

    # ── Resolve instance lists (auto-discover when not provided) ─────────────
    cam1_cotracker = os.path.join(cam1_dir, "cotracker")
    cam2_cotracker = os.path.join(cam2_dir, "cotracker")

    cam1_instances = (
        [i.strip() for i in args.cam1_instances.split(",") if i.strip()]
        if args.cam1_instances
        else discover_instances(cam1_cotracker)
    )
    cam2_instances = (
        [i.strip() for i in args.cam2_instances.split(",") if i.strip()]
        if args.cam2_instances
        else discover_instances(cam2_cotracker)
    )
    if not cam1_instances:
        sys.exit(f"No cam1 instances found in {cam1_cotracker}")
    if not cam2_instances:
        sys.exit(f"No cam2 instances found in {cam2_cotracker}")
    print(f"\ncam1 instances ({len(cam1_instances)}): {cam1_instances}")
    print(f"cam2 instances ({len(cam2_instances)}): {cam2_instances}")

    # ── Load GSAM2 labels per instance ───────────────────────────────────────
    cam1_label_map = get_instance_labels(cam1_dir)
    cam2_label_map = get_instance_labels(cam2_dir)
    if not cam1_label_map:
        print(f"Note: no gsam2 mask metadata in {cam1_dir}; labels will be None")
    if not cam2_label_map:
        print(f"Note: no gsam2 mask metadata in {cam2_dir}; labels will be None")

    # ── Load all cam1 / cam2 cotracker tracks ────────────────────────────────
    print()
    cam1_data = load_instances(cam1_cotracker, cam1_instances, cam1_label_map, "cam1")
    cam2_data = load_instances(cam2_cotracker, cam2_instances, cam2_label_map, "cam2")
    if cam1_data is None:
        sys.exit("No cam1 cotracker files loaded.")
    if cam2_data is None:
        sys.exit("No cam2 cotracker files loaded.")

    tracks1, vis1 = cam1_data["tracks"], cam1_data["vis"]
    tracks2, vis2 = cam2_data["tracks"], cam2_data["vis"]
    if tracks1.shape[0] != tracks2.shape[0]:
        print(
            f"Warning: track length mismatch (cam1 T={tracks1.shape[0]}, "
            f"cam2 T={tracks2.shape[0]}); using min."
        )
    T = min(tracks1.shape[0], tracks2.shape[0])
    print(f"cam1 combined tracks: {tracks1.shape}")
    print(f"cam2 combined tracks: {tracks2.shape}")

    # ── Load MASt3R model ────────────────────────────────────────────────────
    print(f"\nLoading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(device)
    model.eval()

    # ── Result dir (needed for viz and final save) ───────────────────────────
    prefix = args.cam1_name.split("_")[0]
    result_dir = os.path.join(
        args.result_root, prefix, f"{args.cam1_name}__{args.cam2_name}"
    )
    os.makedirs(result_dir, exist_ok=True)

    # ── Match all cam1 × cam2 instance pairs, accumulate pairs ───────────────
    all_idx1, all_idx2 = [], []
    desc_cache = {}  # frame_idx -> (desc1_or_desc2, ...) keyed per call below

    def get_pair_descriptors(qf1, qf2):
        """Run MASt3R for (cam1[qf1], cam2[qf2]) with per-pair caching."""
        key = (qf1, qf2)
        if key in desc_cache:
            return desc_cache[key]
        f1_path = frame_path(cam1_dir, qf1)
        f2_path = frame_path(cam2_dir, qf2)
        images = load_images([f1_path, f2_path], size=args.image_size)
        with torch.no_grad():
            output = inference(
                [tuple(images)], model, device, batch_size=1, verbose=False
            )
        desc1 = output["pred1"]["desc"].squeeze(0)  # (H_m, W_m, D)
        desc2 = output["pred2"]["desc"].squeeze(0)
        desc_cache[key] = (desc1, desc2, f1_path, f2_path)
        return desc_cache[key]

    for i1_idx, inst1 in enumerate(cam1_data["instances"]):
        qf1 = cam1_data["qfs"][i1_idx]
        n1_offset = cam1_data["offsets"][i1_idx]
        cam1_pts_inst = cam1_data["parts"][i1_idx][qf1]  # (N1_inst, 2)
        n1_inst = cam1_pts_inst.shape[0]
        label1 = cam1_data["labels"][i1_idx]

        for i2_idx, inst2 in enumerate(cam2_data["instances"]):
            qf2 = cam2_data["qfs"][i2_idx]
            n2_offset = cam2_data["offsets"][i2_idx]
            cam2_pts_inst = cam2_data["parts"][i2_idx][qf2]  # (N2_inst, 2)
            n2_inst = cam2_pts_inst.shape[0]
            label2 = cam2_data["labels"][i2_idx]

            tag = (
                f"cam1 inst{inst1}({label1})[frame {qf1}] vs "
                f"cam2 inst{inst2}({label2})[frame {qf2}]"
            )

            if not args.ignore_labels and not labels_compatible(label1, label2):
                print(f"\nSkipping {tag}: label mismatch")
                continue

            print(f"\nMatching {tag}...")
            desc1, desc2, f1_path, f2_path = get_pair_descriptors(qf1, qf2)

            feat1 = extract_features_at_points(
                desc1, cam1_pts_inst, orig_w1, orig_h1, args.image_size
            )
            feat2 = extract_features_at_points(
                desc2, cam2_pts_inst, orig_w2, orig_h2, args.image_size
            )

            i1, i2_local, scores = reciprocal_nn(
                feat1.to(device), feat2.to(device), min_score=args.min_score
            )

            n_kept = len(i1)
            mean_score = float(scores.mean()) if n_kept > 0 else 0.0
            print(
                f"  Reciprocal matches: {n_kept} / {min(n1_inst, n2_inst)}"
                f"  mean score: {mean_score:.3f}"
            )

            if n_kept < args.min_matches:
                print(f"  Dropped: {n_kept} < min_matches={args.min_matches}")
                continue
            if mean_score < args.min_pair_mean_score:
                print(
                    f"  Dropped: mean score {mean_score:.3f} "
                    f"< min_pair_mean_score={args.min_pair_mean_score}"
                )
                continue

            i1_global = i1 + n1_offset
            i2_global = i2_local + n2_offset

            if args.visualize:
                viz_path = os.path.join(
                    result_dir,
                    f"matches_cam1_inst{inst1}_vs_cam2_inst{inst2}.jpg",
                )
                visualize_matches(
                    f1_path,
                    f2_path,
                    cam1_pts_inst,
                    cam2_pts_inst,
                    i1,
                    i2_local,
                    viz_path,
                )

            all_idx1.append(i1_global)
            all_idx2.append(i2_global)

    if not any(len(a) for a in all_idx1):
        sys.exit(
            "No matches survived label/score filtering. "
            "Try lowering --min_matches / --min_pair_mean_score, "
            "or pass --ignore_labels."
        )

    # Combine and deduplicate
    idx1_all = np.concatenate(all_idx1)
    idx2_all = np.concatenate(all_idx2)
    pairs = np.unique(np.stack([idx1_all, idx2_all], axis=1), axis=0)
    print(f"\nTotal unique track pairs: {len(pairs)}")

    # ── Seg ids (all class 1 — one dynamic class) ────────────────────────────
    seg_ids1 = np.ones(tracks1.shape[1], dtype=np.int64)
    seg_ids2 = np.ones(tracks2.shape[1], dtype=np.int64)

    # ── Save tracks_match_v2.npz ─────────────────────────────────────────────
    out_path = os.path.join(result_dir, "tracks_match_v2.npz")
    np.savez(
        out_path,
        filtered_corr_indices=pairs,
        tracks1=make_obj_array(
            {"pred_tracks": tracks1, "pred_valid": vis1, "pred_tracks_seg": seg_ids1}
        ),
        tracks2=make_obj_array(
            {"pred_tracks": tracks2, "pred_valid": vis2, "pred_tracks_seg": seg_ids2}
        ),
    )
    print(f"Saved: {out_path}")
    print(f"  filtered_corr_indices: {pairs.shape}")

    # ── VGGT subsets (first T frames) ────────────────────────────────────────
    for cam_name in [args.cam1_name, args.cam2_name]:
        src = os.path.join(args.dataset_root, cam_name, "vggt/camera_parameters.npz")
        if not os.path.exists(src):
            print(f"Warning: {src} not found, skipping VGGT subset for {cam_name}")
            continue
        vg = np.load(src, allow_pickle=True)
        dst = os.path.join(
            args.dataset_root,
            cam_name,
            f"vggt/camera_parameters_{args.vggt_suffix}.npz",
        )
        np.savez(dst, c2w=vg["c2w"][:T], K=vg["K"][:T], valid=vg["valid"][:T])
        print(f"VGGT subset ({T} frames): {dst}")


if __name__ == "__main__":
    main()
