# ContextVAE for UAV Trajectory Prediction (levelXdata)

Adapting [ContextVAE](https://arxiv.org/abs/2302.10873) to UAV-perspective vehicle trajectory prediction on the levelXdata family (inD / uniD / rounD). For the upstream paper README and the original nuScenes/Lyft/Waymo reproduction commands, see [docs/README_upstream.md](docs/README_upstream.md).

This README walks through what's wired up so far: levelXdata preprocessing and the Week 1 smoke test.

## Repo layout (relevant bits)

- [contextvae/](contextvae/) — model code (`main.py`, `model.py`, `data.py`, `utils.py`)
- [configs/](configs/) — training/eval configs (Python files, not YAML)
- [preprocessing/](preprocessing/) — dataset preprocessing scripts + the levelXdata notebook
- [data/levelx/](data/levelx/) — preprocessed ContextVAE-format tensors (gitignored)
- [tmp/](tmp/) — local checkpoint/TensorBoard output dir (gitignored)

Full layout description in [CLAUDE.md](CLAUDE.md).

## Environment

```bash
conda activate new_xupei_env
```

This env carries the CUDA-enabled PyTorch (Python 3.10, PyTorch 1.11). It is the only env from which `contextvae/main.py` should be run. See [docs/xupei_env_notes.md](docs/xupei_env_notes.md) for setup details.

## Preprocessing — inD / uniD / rounD → ContextVAE on-disk format

### Raw data layout

Drop the levelXdata releases under `levelx/` so the recordings sit at:

```
levelx/inD/data/*_tracks.csv          + *_tracksMeta.csv + *_recordingMeta.csv + *_background.png
levelx/uniD/data/*_tracks.csv          + ...
levelx/rounD/data/*_tracks.csv         + ...
```

highD is intentionally excluded — its schema is incompatible (image-frame coordinates, no heading column, no maps shipped). See [docs/levelx_dataset_formats.md](docs/levelx_dataset_formats.md) §2.

### Run the notebook

```bash
jupyter notebook preprocessing/process_levelx.ipynb
```

Execute all cells. The notebook walks `levelx/<dataset>/data/` for inD + uniD + rounD with a single loader, downsamples 25 Hz → 5 Hz, tags vehicles as `VEHICLE/TARGET` and VRUs as `VRU` (kept as context), converts heading deg → rad, and writes ContextVAE on-disk format under `data/levelx/`:

- `train/<dataset>_<recId>.txt` — rows of `fid aid x y heading group`
- `train/<dataset>_<recId>.info` — `<first_frame_id> <map_name>`
- `val/<dataset>_<recId>.{txt,info}` — same format
- `map/<dataset>_<recId>.pkl` — `(semantic_map [3,H,W] in [-1,1], H 3x3)` derived from `XX_background.png`. The 3×3 homography maps local `(x, y)` → image `(row, col)` with the +y-up vs image-y-down flip baked in.

### Verify outputs

```bash
ls data/levelx/train | wc -l       # 110 (current state, 55 recordings × .txt + .info)
ls data/levelx/val   | wc -l       # 30
ls data/levelx/map   | wc -l       # 70
```

### Gotcha: `inclusive_groups`

The on-disk group tag for vehicles is `VEHICLE/TARGET`. The loader at [contextvae/data.py:543](contextvae/data.py#L543) splits the group field on `/`, so configs must filter by either `["TARGET"]` or `["VEHICLE"]`, **not** the literal `["VEHICLE/TARGET"]` (which matches nothing). The provided configs use `["TARGET"]`.

## Smoke test — Week 1 gate

### Configs

- [configs/levelx_train.py](configs/levelx_train.py) — full 100-epoch no-map S-ATTN baseline. OB=10 frames (2 s @ 5 Hz), PRED=25 frames (5 s), OB_RADIUS=30 m, `inclusive_groups=["TARGET"]`, `map_model=None`.
- [configs/levelx_smoke.py](configs/levelx_smoke.py) — 3-line override: `epochs=1`, `test_since=1` so eval runs at end of epoch 1.
- [configs/levelx_eval.py](configs/levelx_eval.py) — eval-side override with `clustering = 5 * pred_samples` for k-means mode selection.

### Pre-flight shape check (optional, ~10 s)

Catches data-format mismatches without spinning up training:

```bash
conda run -n new_xupei_env python -c "
from contextvae.data import Dataloader
dl = Dataloader(files=['data/levelx/train/inD_00.txt'],
    ob_horizon=10, pred_horizon=25, ob_radius=30,
    inclusive_groups=['TARGET'], batch_size=4)
batch = next(iter(dl))
for n, t in zip(['x','y','neighbor'], batch[:3]):
    print(n, getattr(t, 'shape', type(t)))
"
```

Expected output:

```
x (10, 6)
y (25, 2)
neighbor (35, N, 6)
```

`L1=10` observation frames, `L2=25` prediction frames, `L1+L2=35` neighbor-tensor depth.

### Run the smoke train + eval

```bash
conda run -n new_xupei_env python -m contextvae.main \
    --train data/levelx/train/inD_00.txt \
    --test  data/levelx/val/inD_01.txt \
    --config configs/levelx_smoke.py \
    --ckpt tmp/levelx_smoke
```

Outputs land at `tmp/levelx_smoke/`:

- `ckpt-best`, `ckpt-last` — model + optimizer state pickles
- `events.out.tfevents.*` — TensorBoard scalar logs
- `eval_{ADE,FDE}_{deter,min}/` — per-metric event subdirectories

Inspect loss curves:

```bash
tensorboard --logdir tmp/levelx_smoke
```

### Pass criteria

- No exceptions, shape errors, or NaNs across the run.
- Training loss trends downward across the 200 batches (need not be monotonic batch-by-batch; overall trend is what matters).
- ADE/FDE printed at end of epoch 1 land in the rough range of **2–8 m / 5–15 m** for the untrained model. Numbers in the hundreds-of-meters range usually mean the heading-based ego-rotation in `data.py` mis-fired (check that `.txt` headings are radians, not degrees).
- A trained-model floor for comparison comes later — Week 2 runs the full 100-epoch baseline.

## Conventions

- **Output dirs**: use `tmp/<run>` (project-local, gitignored). System `/tmp` is non-persistent on this machine.
- **Configs are plain Python.** Relative imports (`from .levelx_train import *`) work because [contextvae/main.py:35](contextvae/main.py#L35) registers the loaded config under `sys.modules[spec.name]` before exec.

## Pointers

- [docs/thesis_plan.md](docs/thesis_plan.md) — full 7-week thesis schedule, locked decisions, risk register.
- [docs/levelx_dataset_formats.md](docs/levelx_dataset_formats.md) — column-by-column reference for every levelXdata file format.
- [docs/contextvae_training_pipeline.md](docs/contextvae_training_pipeline.md) — model + training pipeline notes.
- [docs/README_upstream.md](docs/README_upstream.md) — original ContextVAE paper README + nuScenes/Lyft/Waymo reproduction commands.
