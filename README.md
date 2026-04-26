<br />
<p align="center">

  <h1 align="center">VisualSync: Multi-Camera Synchronization via Cross-View Object Motion</h1>

  <p align="center">
   NeurIPS, 2025
    <br />
    <a href="https://stevenlsw.github.io"><strong>Shaowei Liu*</strong></a>
    ·
    <a href="https://Davidyao99.github.io/"><strong>David Yifan Yao*</strong></a>
    ·
    <a href="https://saurabhg.web.illinois.edu/"><strong>Saurabh Gupta†</strong></a>
    ·
    <a href="https://shenlong.web.illinois.edu/"><strong>Shenlong Wang†</strong></a>
    ·
  </p>

  <p align="center">
    <a href='https://drive.google.com/file/d/1-MwQRWBm_I3576gBaC_f7D3iYcFpiox0/view?usp=sharing'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'></a>
    <a href='https://arxiv.org/abs/2512.02017'><img src='https://img.shields.io/badge/arXiv-2512.02017-b31b1b.svg'  alt='Arxiv'></a>
    <a href='https://stevenlsw.github.io/visualsync/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'></a>
  </p>

</p>
<br />

This repository contains the PyTorch implementation for [VisualSync: Multi-Camera Synchronization via Cross-View Object Motion](https://stevenlsw.github.io/visualsync/), NeurIPS 2025. **VisualSync** aligns unsynchronized multi-view videos by matching object motion with epipolar cues.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/stevenlsw/visualsync.git
cd visualsync
bash scripts/install.sh   # creates conda env 'visualsync'
conda activate visualsync
```

> Tested with CUDA 12.4.

## Data format

Place raw videos under `raw_data/<scene>/` as `.mp4` files, one per camera. All other directories are created automatically by the pipeline.

```
raw_data/
  scene1/
    cam1.mp4
    cam2.mp4
    ...
data/                              ← created automatically
  scene1/
    scene1_cam1/
      rgb/                         ← extracted frames ('extract')
      gpt_video/                   ← dynamic object tags ('tags')
      gsam2/
        mask/                      ← segmentation masks ('sam2')
        vis/                       ← segmentation visualizations ('sam2')
      vggt/                        ← per-cam camera parameters ('vggt')
      cotracker/                   ← pixel-level tracks ('cotracker')
      mast3r/                      ← cross-cam correspondences ('match')
    scene1_cam2/
      ...
    vggt_output/                   ← COLMAP-format pose export ('vggt')
    videos/                        ← merged mp4s for CoTracker input ('merge')
results/                           ← created automatically
  scene1/
    scene1_cam1__scene1_cam2/      ← sync outputs ('sync')
```

## Running the pipeline

The end-to-end pipeline is driven by a single script:

```bash
python scripts/run_pipeline.py --scene scene1
```

This runs all steps in order:

| Step | Name | What it does |
|------|------|--------------|
| 0 | `weights` | Download SAM2 / MASt3R / VGGT checkpoints |
| 1 | `extract` | Extract frames from `.mp4` files into `data/<scene>/<scene>_camN/rgb/` |
| 2 | `tags` | Write `gpt_video/tags.json` (dynamic object labels) into each cam dir |
| 3 | `sam2` | Run Grounded-SAM2 to segment dynamic objects per frame |
| 4 | `vggt` | Estimate camera poses with VGGT, export as COLMAP |
| 5 | `merge` | Merge rgb/mask frames into `.mp4` videos for CoTracker |
| 6 | `cotracker` | Run CoTracker per camera to get pixel-level tracks |
| 7 | `match` | Run MASt3R to establish cross-cam correspondences |
| 8 | `sync` | Estimate temporal sync offsets per camera pair |

## Options

### Required

| Flag | Description |
|------|-------------|
| `--scene` | Scene name, e.g. `scene1` |

### Paths

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/` | Root directory for input data |
| `--results-dir` | `results/` | Root directory for outputs |

### Step control

| Flag | Description |
|------|-------------|
| `--skip weights,extract,...` | Skip one or more steps (comma-separated) |
| `--only match,sync` | Run only the specified steps, skip everything else |

Valid step names: `weights`, `extract`, `tags`, `sam2`, `vggt`, `merge`, `cotracker`, `match`, `sync`.

### Dynamic object tags

| Flag | Default | Description |
|------|---------|-------------|
| `--tags-file` | built-in | Path to a JSON file with `{"dynamic": ["label1", ...]}` broadcast to every cam |

The built-in default tags are: `man`, `woman`, `pipette`, `hand`.

### CoTracker

| Flag | Default | Description |
|------|---------|-------------|
| `--cotracker-grid-size` | `100` | Grid size for track seeding |
| `--cotracker-model-type` | `offline` | `online` or `offline` |
| `--cotracker-device` | `auto` | `auto` (cuda > mps > cpu), `cuda`, `mps`, or `cpu` |
| `--cotracker-cpu-fallback` / `--no-cotracker-cpu-fallback` | enabled | Retry on CPU if CUDA OOMs |

### MASt3R matching & sync

| Flag | Default | Description |
|------|---------|-------------|
| `--vggt-suffix` | `300` | Frame count suffix for `camera_parameters_<suffix>.npz` |
| `--offset-range` | `150` | Max frame offset to search during sync |
| `--pairs` | all pairs | Cam-index pairs to match/sync, e.g. `1-2,1-3,2-3` |

## Examples

Run everything for a new scene:
```bash
python scripts/run_pipeline.py --scene scene1
```

Skip weight download and frame extraction if already done:
```bash
python scripts/run_pipeline.py --scene scene1 --skip weights,extract
```

Re-run only the matching and sync steps:
```bash
python scripts/run_pipeline.py --scene scene1 --only match,sync
```

Run on a specific subset of camera pairs:
```bash
python scripts/run_pipeline.py --scene scene1 --only match,sync --pairs 1-2,1-3
```

Use custom dynamic object labels:
```bash
python scripts/run_pipeline.py --scene scene1 --tags-file my_tags.json
```

## Citation

```bibtex
@inproceedings{liu2025visualsync,
  title={VisualSync: Multi-Camera Synchronization via Cross-View Object Motion},
  author={Liu, Shaowei and Yao, David Yifan and Gupta, Saurabh and Wang, Shenlong},
  booktitle={NeurIPS},
  year={2025}
}
```

## Acknowledgements

- [SAM2](https://github.com/facebookresearch/sam2) for video segmentation
- [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/) for object tracking
- [CoTracker3](https://github.com/facebookresearch/co-tracker) for pixel-level tracking
- [VGGT](https://github.com/facebookresearch/vggt) for camera pose estimation
- [MASt3R](https://github.com/naver/mast3r) for cross-view correspondence
- [Uni4D](https://github.com/Davidyao99/uni4d/) for dynamic object segmentation
