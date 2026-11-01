# ContextVAE for UAV Trajectory Prediction (levelXdata)

Adaptation of [ContextVAE](https://arxiv.org/abs/2302.10873) (Xu, Hayet & Karamouzas, IEEE RA-L / ICRA 2024) to **vehicle trajectory prediction from a hovering-UAV perspective**, trained and evaluated on the [levelXdata](https://levelxdata.com) family (inD + uniD + rounD).

The original nuScenes / Lyft / Waymo model and reproduction commands come from the upstream repository: <https://github.com/xupei0610/ContextVAE>.

## Overview

This work replaces the LSTM trajectory predictor in **Ntousis (2024)** — a dual-stage UAV detection-and-tracking system (onboard RPi 4B + NCS2; server-side A40 with YOLOv8x + DeepSORT + LSTM) — with a ContextVAE-style timewise CVAE adapted to the aerial-drone domain.

Four concrete contributions:

1. **Multi-modal stochastic prediction** in place of the unimodal LSTM — a timewise latent `z_t`, K-sample output, evaluated by `minADE_k` / `minFDE_k` rather than RMSE.
2. **S-ATTN social encoder** with radius-based neighbor selection over DeepSORT tracks (replaces the original fixed 4-direction neighbor concatenation).
3. **Aerial-orthomap M-ATTN encoder** — the main architectural contribution. ContextVAE's HD-map raster CNN is swapped for a ResNet-18 (ImageNet-pretrained) over heading-rotated, agent-centered crops of the per-recording orthomap (`XX_background.png`). UAV deployments have no HD vector maps but do have geo-referenced orthoimagery.
4. **Rolling-window streaming inference** integrated into the server-side socket pipeline.

Primary training set: inD + uniD + rounD (drone-recorded bird's-eye view, metric SI coordinates). highD is excluded by design — image-frame coordinates, no headings, no maps.

## Repository layout

```
ContextVAE/
├── contextvae/                   # Python package
│   ├── main.py                   # entry point: train + eval loop
│   ├── model.py                  # ContextVAE module (S-ATTN + M-ATTN + timewise CVAE)
│   ├── model_baseline.py         # vanilla ContextVAE reference (no thesis edits)
│   ├── data.py                   # Dataloader + FixedNumberBatchSampler + map augmentation
│   ├── lstm_baseline.py          # port of the Ntousis LSTM, on ContextVAE's data interface
│   ├── utils.py                  # ADE_FDE, k-means clustering, seeding, RNG helpers
│   ├── eval_perseq.py            # per-sequence cross-domain ADE/FDE (micro + macro)
│   ├── plot_*.py                 # figure generation (metrics, architecture, qualitative, results)
│   └── visualize.py              # per-agent orthomap + K-sample inference plots
├── configs/                      # plain-Python configs (not YAML)
│   ├── levelx_train.py           # full 60-epoch M-ATTN headline config
│   ├── levelx_train_vehicle_only.py  # vehicle-only ablation (drops VRU neighbors)
│   ├── levelx_eval.py            # eval-side overrides (clustering = 5 × pred_samples)
│   ├── levelx_smoke{,_map,_nomap}.py  # 1-epoch / 20-batch smoke tests
│   ├── levelx_mini.py            # 5-epoch / 200-batch pre-flight before the headline run
│   ├── visdrone_eval_{2,3,5}s.py # cross-domain VisDrone eval at 2/3/5 s horizons
│   └── {nuscenes,lyft,waymo}_{train,eval}.py  # upstream reproductions
├── preprocessing/
│   ├── process_levelx.ipynb      # levelXdata → ContextVAE on-disk format
│   └── process_{nuscenes,lyft,waymo,visdrone}.py  # other dataset preprocessors
├── data/levelx/                  # preprocessed tensors (gitignored)
│   ├── train/  val/              # `<dataset>_<recId>.{txt,info}` pairs
│   └── map/                      # `<dataset>_<recId>.pkl` orthomap + homography
├── levelx/                       # raw levelXdata CSVs + backgrounds (gitignored)
│   └── inD/data/   uniD/data/   rounD/data/
├── tests/                        # smoke + map-verification scripts
├── tmp/                          # checkpoints + TensorBoard logs (gitignored)
└── uav_guidance/                 # server-side integration of ContextVAE into the UAV pipeline
```

## Environment

```bash
conda create --name uav-traj --file requirements.txt -c pytorch -c conda-forge
conda activate uav-traj
```

Python 3.10, PyTorch 1.11 (CUDA-enabled), NumPy 1.21, TorchVision, TensorBoard. All `contextvae/main.py` runs use this environment.

> A separate `waymo` env (TensorFlow 2.6, protobuf <3.20) is needed only for `preprocessing/process_waymo.py`; the levelXdata path does not use it.

## Preprocessing — levelXdata → ContextVAE tensors

### Raw data layout

Place the levelXdata releases under `levelx/` so the recordings sit at:

```
levelx/inD/data/*_tracks.csv    + *_tracksMeta.csv + *_recordingMeta.csv + *_background.png
levelx/uniD/data/*_tracks.csv   + ...
levelx/rounD/data/*_tracks.csv  + ...
```

### Run the notebook

```bash
jupyter notebook preprocessing/process_levelx.ipynb
```

Execute all cells. The unified loader walks `levelx/<dataset>/data/` for inD + uniD + rounD, downsamples 25 Hz → 5 Hz, tags vehicles as `VEHICLE/TARGET` and VRUs as `VRU` (kept as context), converts heading deg → rad, and writes:

- `train/<dataset>_<recId>.txt` — rows of `fid aid x y heading group`
- `train/<dataset>_<recId>.info` — `<first_frame_id> <map_name>`
- `val/<dataset>_<recId>.{txt,info}` — same format
- `map/<dataset>_<recId>.pkl` — `(semantic_map [3,H,W] in [-1,1], H 3×3)` built from `XX_background.png`. The 3×3 homography maps local `(x, y)` → image `(row, col)` with the +y-up vs image-y-down flip baked in, and the raster is pre-padded by `EXT_PAD` so every agent-centered crop is full-size.

### Verify outputs

```bash
ls data/levelx/train | wc -l     # 110  (55 recordings × .txt + .info)
ls data/levelx/val   | wc -l     # 30
ls data/levelx/map   | wc -l     # 70
```

Aggregate train-split size: ~1M sliding-window trajectory samples across 28,207 unique agents.

### Pre-flight shape check (~10 s)

```bash
python -c "
from contextvae.data import Dataloader
dl = Dataloader(files=['data/levelx/train/inD_00.txt'],
    ob_horizon=10, pred_horizon=25, ob_radius=30,
    inclusive_groups=['TARGET'], batch_size=4)
batch = next(iter(dl))
for n, t in zip(['x','y','neighbor'], batch[:3]):
    print(n, getattr(t, 'shape', type(t)))
"
```

Expected: `x (10, 6)`, `y (25, 2)`, `neighbor (35, N, 6)`. `L1 + L2 = 35` is the full observation + prediction neighbor window the posterior RNN consumes.

### Loader gotcha: `inclusive_groups`

The on-disk group tag for vehicles is `VEHICLE/TARGET`. The loader in [contextvae/data.py](contextvae/data.py) splits on `/`, so configs must filter by `["TARGET"]` or `["VEHICLE"]`, **not** the literal `["VEHICLE/TARGET"]` (which matches nothing). All shipped configs use `["TARGET"]`.

## Training and evaluation

All commands run from the repo root inside the `uav-traj` environment.

### Headline run — M-ATTN (orthomap variant)

```bash
python -m contextvae.main \
    --train   data/levelx/train \
    --test    data/levelx/val \
    --map_dir data/levelx/map \
    --config  configs/levelx_train.py \
    --ckpt    tmp/levelx_full_map_v1
```

[configs/levelx_train.py](configs/levelx_train.py) defaults: 60 epochs, OB = 10 frames (2 s @ 5 Hz), PRED = 25 frames (5 s), OB_RADIUS = 30 m, batch = 256, `batches_per_epoch = 1000` (~25 % of the train split), `inclusive_groups = ["TARGET"]`, `map_model = "resnet18"` (ImageNet weights, dropout p = 0.15), training-time map augmentation on (colour jitter ±0.2, rotation ±5°, scale ±10%). Wall-clock ≈ 4–8 h on a single A40.

### No-map S-ATTN baseline

Same command, omit `--map_dir` (or set `model["map_model"] = None` in the config). The no-map run is the defensible primary if M-ATTN does not outperform it — the ContextVAE supplementary (Table S3) reports it at ~95 % of full-model performance.

### Smoke tests (~1–2 min on A40)

Sanity-check the full pipeline end-to-end before any multi-hour run:

```bash
# No-map baseline smoke
python -m contextvae.main \
    --train  data/levelx/train --test data/levelx/val \
    --config configs/levelx_smoke_nomap.py \
    --ckpt   tmp/smoke_nomap

# M-ATTN orthomap smoke (needs --map_dir)
python -m contextvae.main \
    --train  data/levelx/train --test data/levelx/val \
    --map_dir data/levelx/map \
    --config configs/levelx_smoke_map.py \
    --ckpt   tmp/smoke_map
```

The smoke configs inherit from `levelx_train.py` and trim epochs/batches.

### Mini pre-flight (~20 min)

[configs/levelx_mini.py](configs/levelx_mini.py) — 5 epochs × 200 batches with full M-ATTN. Surfaces OOM, dataloader bottlenecks, posterior collapse, and eval-loop regressions before committing to the 4–8 h headline run.

### Eval-only against an existing checkpoint

```bash
python -m contextvae.main \
    --test    data/levelx/val \
    --map_dir data/levelx/map \
    --config  configs/levelx_eval.py \
    --ckpt    tmp/levelx_full_map_v1
```

[configs/levelx_eval.py](configs/levelx_eval.py) sets `clustering = 5 * pred_samples`, so eval uses k-means mode selection over an oversampled prediction pool. For per-sequence cross-domain evaluation (VisDrone), see [contextvae/eval_perseq.py](contextvae/eval_perseq.py).

### Outputs

Each `--ckpt` directory contains:

- `ckpt-best`, `ckpt-last` — model + optimizer state
- `events.out.tfevents.*` — TensorBoard scalars (`train/loss`, `train/kl`)
- `eval_{ADE,FDE}_{deter,min}/` — per-metric event subdirectories

## Visualization and analysis

### Training/eval curves

[contextvae/plot_metrics.py](contextvae/plot_metrics.py) renders a 2×2 PNG: `train/loss` (log), `train/kl` (posterior-collapse check), `eval_ADE`, `eval_FDE` (min-of-K solid, deterministic dotted), optionally overlaying a baseline run as reference lines.

```bash
# Single run
python -m contextvae.plot_metrics tmp/levelx_full_map_v1

# Overlay a no-map baseline
python -m contextvae.plot_metrics tmp/levelx_full_map_v1 \
    --baseline tmp/nomap-s-attn \
    --out      tmp/levelx_full_map_v1/metrics_vs_baseline.png \
    --title    "M-ATTN headline vs no-map baseline"
```

### Per-agent inference visualizations

[contextvae/visualize.py](contextvae/visualize.py) iterates the first `--num_batches` eval batches, runs `model(..., n_predictions=K)`, and writes one PNG per agent with orthomap + observation history + ground-truth future + K predicted samples in agent-meter coordinates.

```bash
python -m contextvae.visualize \
    --config  configs/levelx_eval.py \
    --ckpt    tmp/levelx_full_map_v1/ckpt-best \
    --test    data/levelx/val \
    --map_dir data/levelx/map \
    --output  tmp/levelx_full_map_v1/viz \
    --num_batches 20 --n_predictions 5
```

### TensorBoard

```bash
tensorboard --logdir tmp/levelx_full_map_v1
```

## UAV pipeline integration

`uav_guidance/` holds the server-side integration that swaps the ContextVAE predictor into the dual-stage UAV pipeline, under `server_code_only/main_program/`:

- `contextvae_inference.py` — loads a checkpoint + map raster once and runs single-target streaming inference per frame, mirroring the training loader's localization, neighbor padding, and map crop.
- `pixel_metric_shim.py` / `ego_motion_shim.py` — convert between image pixels and ContextVAE metric-world coordinates (fixed homography for a stationary drone; ORB + RANSAC ego-motion compensation for a moving one).
- `scene_recog_socket_yolov8.py` — the server loop (YOLOv8 + DeepSORT) with the LSTM predictor replaced by ContextVAE.
- `replay_video.py`, `synth_clip_gen.py`, `verify_inference.py`, `visual_sanity.py` — offline replay, synthetic-clip generation, and verification utilities.

## Conventions

- **Output directories**: project-local `tmp/<run>` (gitignored). Checkpoints and logs live here.
- **Configs are plain Python**, loaded with `importlib.util`. Relative imports (`from .levelx_train import *`) work because `contextvae/main.py` registers the loaded config under `sys.modules[spec.name]` before executing it, with `submodule_search_locations` pointing at `configs/`.
- **Velocity units**: the 6-D state's `vx, vy` are in metres per timestep (0.2 s @ 5 Hz), **not** m/s. Multiply by 5 before comparing against an m/s baseline.
- **Neighbor padding** is `1e9`, not 0 — the radius mask in `enc()` filters by `dist <= ob_radius`, and zero-padding would erroneously pass the threshold.

## Acknowledgements

- **ContextVAE** — Pei Xu, Jean-Bernard Hayet, Ioannis Karamouzas, *Context-aware Timewise VAEs for Real-time Vehicle Trajectory Prediction*, IEEE RA-L 2023 / ICRA 2024. Upstream code: <https://github.com/xupei0610/ContextVAE>.
- **Dual-stage UAV detection-and-tracking system** — Odysseas Ntousis (2024); `uav_guidance/` builds on that codebase.
- **levelXdata** (inD, uniD, rounD) — ika, RWTH Aachen University: <https://levelxdata.com>.
