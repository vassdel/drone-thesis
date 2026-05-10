"""Render per-epoch metric curves from a ContextVAE training run.

Reads the TensorBoard event files written by `contextvae.main` and emits a
single 2x2 PNG. The four panels are:

    top-left   train/loss (log scale)
    top-right  train/kl  (KL active <=> non-zero => no posterior collapse)
    bottom-left   eval_ADE (min-of-K solid line; deterministic dotted)
    bottom-right  eval_FDE (min-of-K solid line; deterministic dotted)

Run inside the project conda env (`new_xupei_env`), e.g.
`conda run --no-capture-output -n new_xupei_env python -m contextvae.plot_metrics ...`.

Usage examples
--------------
1) Default render on a single run dir (writes <run_dir>/metrics.png):

       python -m contextvae.plot_metrics tmp/nomap-s-attn

2) Overlay another run's final eval values as horizontal dashed reference
   lines on the ADE/FDE panels:

       python -m contextvae.plot_metrics tmp/levelx_full_map_v1 \\
           --baseline tmp/nomap-s-attn

3) Custom output path and figure suptitle:

       python -m contextvae.plot_metrics tmp/levelx_full_map_v1 \\
           --baseline tmp/nomap-s-attn \\
           --out tmp/levelx_full_map_v1/metrics_vs_baseline.png \\
           --title "M-ATTN headline vs no-map baseline"

CLI options
-----------
run_dir              (positional, required) Directory containing TB event
                     files. The root event file supplies `train/*` tags;
                     each subdir (e.g. `eval_ADE_min/`) is read separately
                     and labelled by its name (the writers in main.py log
                     under the bare tag `eval`, so the subdir name is the
                     only disambiguator).
--baseline DIR       (optional) A second run dir. Its final-epoch
                     `eval_ADE_min` and `eval_FDE_min` values are drawn as
                     dashed horizontal lines on the ADE/FDE panels for
                     quick "did we beat the baseline" comparison.
--out PATH           (optional) Output PNG path.
                     Default: <run_dir>/metrics.png.
--title STR          (optional) Figure suptitle.
                     Default: basename of run_dir.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_named(d):
    out = {}
    for root, _, files in os.walk(d):
        if not any("tfevents" in f for f in files):
            continue
        ea = EventAccumulator(root, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if root == d:
            for tag in tags:
                out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
        elif tags:
            label = os.path.basename(root)
            out[label] = [(e.step, e.value) for e in ea.Scalars(tags[0])]
    return out


def series(src, label):
    if label in src and src[label]:
        s, v = zip(*src[label])
        return list(s), list(v)
    return [], []


def render(run_dir, baseline_dir=None, out=None, title=None):
    run = load_named(run_dir)
    base = load_named(baseline_dir) if baseline_dir else {}

    base_ade = base["eval_ADE_min"][-1][1] if "eval_ADE_min" in base else None
    base_fde = base["eval_FDE_min"][-1][1] if "eval_FDE_min" in base else None

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    xs, ys = series(run, "train/loss")
    ax[0, 0].plot(xs, ys, color="C0")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("train/loss (log)")
    ax[0, 0].set_xlabel("epoch")
    ax[0, 0].grid(alpha=0.3)

    xs, ys = series(run, "train/kl")
    ax[0, 1].plot(xs, ys, color="C0")
    ax[0, 1].set_title("train/kl")
    ax[0, 1].set_xlabel("epoch")
    ax[0, 1].grid(alpha=0.3)

    xs, ys = series(run, "eval_ADE_min")
    xs_d, ys_d = series(run, "eval_ADE_deter")
    ax[1, 0].plot(xs, ys, color="C0", marker="o", label="min-of-K")
    if xs_d:
        ax[1, 0].plot(xs_d, ys_d, color="C0", linestyle=":", marker=".", alpha=0.6,
                      label="deterministic")
    if base_ade is not None:
        ax[1, 0].axhline(base_ade, color="C3", linestyle="--",
                         label=f"baseline ({base_ade:.4f})")
    ax[1, 0].set_title("eval_ADE")
    ax[1, 0].set_xlabel("epoch")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend()

    xs, ys = series(run, "eval_FDE_min")
    xs_d, ys_d = series(run, "eval_FDE_deter")
    ax[1, 1].plot(xs, ys, color="C0", marker="o", label="min-of-K")
    if xs_d:
        ax[1, 1].plot(xs_d, ys_d, color="C0", linestyle=":", marker=".", alpha=0.6,
                      label="deterministic")
    if base_fde is not None:
        ax[1, 1].axhline(base_fde, color="C3", linestyle="--",
                         label=f"baseline ({base_fde:.4f})")
    ax[1, 1].set_title("eval_FDE")
    ax[1, 1].set_xlabel("epoch")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend()

    plt.suptitle(title or os.path.basename(os.path.normpath(run_dir)),
                 fontsize=13, y=1.00)
    plt.tight_layout()

    out_path = out or os.path.join(run_dir, "metrics.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Render train+eval scalars from a TB logdir to a single PNG."
    )
    p.add_argument("run_dir", help="Run directory containing TB event files")
    p.add_argument("--baseline", default=None,
                   help="Optional second run dir; its final eval values become "
                        "dashed reference lines")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <run_dir>/metrics.png)")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    render(args.run_dir, args.baseline, args.out, args.title)


if __name__ == "__main__":
    main()
