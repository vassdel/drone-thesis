"""
eval_perseq.py — per-sequence cross-domain eval with micro + macro aggregates.

Motivation. `contextvae/main.py` pools every test trajectory into one loader and
reports a single window-weighted (micro) ADE/FDE. On VisDrone that lets a few
long sequences dominate (e.g. one seq = 65 % of the 5 s windows), so the pooled
number reads as "one intersection," not a cross-domain average. This script runs
the SAME eval math as main.py but PER SEQUENCE, then reports:

  - a per-sequence table,
  - MICRO average  (window-weighted; reproduces main.py's pooled number),
  - MACRO average  (mean over sequences -> the robust "typical scene" headline)
                   with std / min / max across scenes,
  - a constant-velocity (CV) baseline per sequence (scale-robust sanity).

The clustered min-of-K path is copied verbatim from main.py:246-262 (no-map
branch) so the numbers are directly comparable. RNG is reset to the same initial
state main.py uses (seed -> get_rng_state -> set_rng_state before each seq), so
the DETERMINISTIC micro-average matches main.py exactly and minADE matches
closely (batch composition differs across the seq boundary, which slightly
perturbs the stochastic sampling order — expected).

Usage (identical args to main.py eval; no --map_dir = no-map path):
    PY=/home/vdelis/anaconda3/envs/new_xupei_env/bin/python
    $PY -m contextvae.eval_perseq --test data/visdrone_levelx/val \
        --config configs/visdrone_eval_2s.py --ckpt tmp/nomap-s-attn/ckpt-best
"""

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import torch

from contextvae.model import ContextVAE
from contextvae.data import Dataloader
from contextvae.utils import ADE_FDE, clustering, seed, get_rng_state, set_rng_state


def load_config(path):
    spec = importlib.util.spec_from_file_location(
        "config", path, submodule_search_locations=[os.path.dirname(path)])
    cfg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cfg
    spec.loader.exec_module(cfg)
    return cfg


def build_model(config, ckpt_path, device):
    model_cls = getattr(config, "model_cls", ContextVAE)
    model = model_cls(**config.model).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    model.eval()
    return model, state_dict


def predict_clustered(model, x, neighbor, config):
    """min-of-K with clustering — verbatim from main.py:246-262 (no-map branch)."""
    if config.clustering:
        if getattr(model, "use_map", False):
            y_ = model(x, neighbor, n_predictions=config.clustering)
        else:
            reps = int(np.ceil(config.clustering / config.pred_samples))
            y_ = torch.cat(
                [model(x, neighbor, n_predictions=config.pred_samples) for _ in range(reps)], 0)
        cand = []
        for i in range(y_.size(-2)):
            traj, _ = clustering(y_[..., i, :].cpu().numpy(), n_samples=config.pred_samples)
            cand.append(traj)
        y_ = torch.as_tensor(np.stack(cand, 2), device=y_.device, dtype=y_.dtype)
    else:
        y_ = model(x, neighbor, n_predictions=config.pred_samples)
    return y_  # n_samples x L x N x 2


def cv_error(x, y):
    """Constant-velocity ADE/FDE per window. x:(L,N,6) y:(H,N,2). Returns (ade(N,), fde(N,))."""
    last_pos = x[-1, :, :2]
    last_vel = x[-1, :, 2:4]
    H = y.shape[0]
    steps = torch.arange(1, H + 1, device=y.device, dtype=y.dtype).view(H, 1, 1)
    cv = last_pos.unsqueeze(0) + steps * last_vel.unsqueeze(0)
    err = (cv - y).norm(dim=-1)  # (H, N)
    return err.mean(0), err[-1]


def eval_one_file(path, config, model, device, init_rng):
    """Run the full eval on a single .txt; return per-window arrays, or None if empty."""
    try:
        ds = Dataloader(
            [path],
            batch_first=False, device=device, seed=1,
            **config.test_dataloader,
            map_dir=None, shuffle=False,
        )
    except Exception:
        return None  # 0-trajectory sequence (e.g. pedestrian-only) -> loader has no data
    if len(ds.data) == 0:
        return None

    loader = torch.utils.data.DataLoader(
        ds, collate_fn=ds.collate_fn, batch_sampler=ds.batch_sampler)

    set_rng_state(init_rng, device)  # reproducible per-seq stochastic sampling
    ade_min, fde_min, ade_det, fde_det, ade_cv, fde_cv = [], [], [], [], [], []
    with torch.no_grad():
        for item in loader:
            item = [t.to(device) if torch.is_tensor(t) else t for t in item]
            x, y, neighbor, *_ = item
            y_ = predict_clustered(model, x, neighbor, config)
            a, f = ADE_FDE(y_, y)
            ade_min.append(a.min(0)[0]); fde_min.append(f.min(0)[0])
            yd = model(x, neighbor, n_predictions=0)
            a, f = ADE_FDE(yd, y)
            ade_det.append(a); fde_det.append(f)
            a, f = cv_error(x, y)
            ade_cv.append(a); fde_cv.append(f)
    cat = lambda L: torch.cat(L).cpu().numpy()
    return dict(ade_min=cat(ade_min), fde_min=cat(fde_min),
                ade_det=cat(ade_det), fde_det=cat(fde_det),
                ade_cv=cat(ade_cv), fde_cv=cat(fde_cv))


def main():
    ap = argparse.ArgumentParser(description="Per-sequence cross-domain eval (micro+macro).")
    ap.add_argument("--test", required=True, help="Dir of <seq>.txt files.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    seed(args.seed)
    init_rng = get_rng_state(device)
    model, sd = build_model(config, args.ckpt, device)
    print(f"ckpt epoch={sd.get('epoch')}  in-dist eval(min/det): "
          f"ADE {sd['ade']:.4f}/{sd['ade_d']:.4f}  FDE {sd['fde']:.4f}/{sd['fde_d']:.4f}")
    print(f"config={args.config}  pred_horizon={config.test_dataloader['pred_horizon']}  "
          f"clustering={config.clustering}  pred_samples={config.pred_samples}\n")

    files = sorted(glob.glob(os.path.join(args.test, "*.txt")))
    per_seq = []
    print(f"{'sequence':24s} {'win':>6s}  {'minADE':>7s} {'detADE':>7s} {'cvADE':>7s}  "
          f"{'minFDE':>7s} {'detFDE':>7s} {'cvFDE':>7s}")
    pooled = {k: [] for k in ("ade_min", "fde_min", "ade_det", "fde_det", "ade_cv", "fde_cv")}
    for f in files:
        name = os.path.basename(f).replace(".txt", "")
        r = eval_one_file(f, config, model, device, init_rng)
        if r is None:
            print(f"{name:24s} {'0':>6s}  (no vehicle targets — skipped)")
            continue
        for k in pooled:
            pooled[k].append(r[k])
        n = len(r["ade_min"])
        means = {k: float(r[k].mean()) for k in pooled}
        per_seq.append((name, n, means))
        print(f"{name:24s} {n:6d}  {means['ade_min']:7.3f} {means['ade_det']:7.3f} "
              f"{means['ade_cv']:7.3f}  {means['fde_min']:7.3f} {means['fde_det']:7.3f} "
              f"{means['fde_cv']:7.3f}")

    if not per_seq:
        print("\nNo sequences with vehicle targets.")
        return 0

    # MICRO = window-weighted (== main.py pooled). MACRO = mean over sequences.
    micro = {k: float(np.concatenate(pooled[k]).mean()) for k in pooled}
    macro = {k: np.array([m[k] for _, _, m in per_seq]) for k in pooled}
    n_seq = len(per_seq)
    tot_win = sum(n for _, n, _ in per_seq)
    print("\n" + "=" * 78)
    print(f"sequences with targets: {n_seq}   total windows: {tot_win}")
    print(f"MICRO (window-weighted, == main.py pooled):")
    print(f"   minADE {micro['ade_min']:.4f}  detADE {micro['ade_det']:.4f}  "
          f"cvADE {micro['ade_cv']:.4f}  |  minFDE {micro['fde_min']:.4f}  "
          f"detFDE {micro['fde_det']:.4f}  cvFDE {micro['fde_cv']:.4f}")
    print(f"MACRO (per-sequence mean +/- std  [min, max], n={n_seq} scenes):")
    for label, kmin in [("minADE", "ade_min"), ("detADE", "ade_det"), ("cvADE", "ade_cv"),
                        ("minFDE", "fde_min"), ("detFDE", "fde_det"), ("cvFDE", "fde_cv")]:
        a = macro[kmin]
        print(f"   {label:7s} {a.mean():.4f} +/- {a.std():.4f}  [{a.min():.4f}, {a.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
