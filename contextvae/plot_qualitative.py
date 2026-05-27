"""Compose qualitative thesis figures from existing replay frames.

Generates the following PNGs into ``--out-dir`` (default ``tmp/figs``):

- ``qual_visdrone_pair.png``     — VisDrone aid5 stationary vs ego-motion (B3)
- ``qual_failure_egomotion.png`` — synthetic stationary drift vs ego-motion fix (B5)

B1 (multi-modality) is composed by ``compose_multimodal`` once
``contextvae.visualize`` has written K=5 per-agent overlays.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def _imshow_with_title(ax, path, title):
    img = Image.open(path)
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def compose_pair(left_path, right_path, left_title, right_title,
                 figure_title, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    _imshow_with_title(ax[0], left_path, left_title)
    _imshow_with_title(ax[1], right_path, right_title)
    fig.suptitle(figure_title, fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def compose_visdrone(out_dir):
    compose_pair(
        left_path="tmp/replay_frames/replay_visdrone_gt_stationary_aid5_f090_crop.png",
        right_path="tmp/replay_frames/replay_visdrone_gt_egomotion_aid5_f090_crop.png",
        left_title="(a) Stationary shim (no ego-motion comp.)",
        right_title="(b) MovingCameraShim (ORB+MAGSAC, re-anchor)",
        figure_title=(
            "VisDrone cross-domain — uav0000305_00000_v, agent 5, frame 90. "
            "Blue: obs (10 frames). Red: M-ATTN K=5 mean prediction (6 frames)."
        ),
        out_path=os.path.join(out_dir, "qual_visdrone_pair.png"),
    )


def compose_failure_egomotion(out_dir):
    compose_pair(
        left_path="tmp/replay_frames/replay_synth_stationary_f075.png",
        right_path="tmp/replay_frames/replay_synth_egomotion_f075.png",
        left_title="(a) Without ego-motion comp. (DRIFT)",
        right_title="(b) With MovingCameraShim (corrected)",
        figure_title=(
            "Failure mode: stationary-shim drift on synthetic moving-camera clip. "
            "Frame 75/149. Red banner in (a) flags the expected drift; "
            "(b) recovers via per-frame homography to keyframe K0."
        ),
        out_path=os.path.join(out_dir, "qual_failure_egomotion.png"),
    )


def compose_multimodal(viz_dir, out_dir, n_panels=4):
    """Compose a 2x2 panel from K=5 overlay PNGs produced by contextvae.visualize."""
    cand = sorted(glob.glob(os.path.join(viz_dir, "*.png")))
    if not cand:
        print(f"WARN: no PNGs found under {viz_dir}; skipping B1")
        return None
    # Pick spread-out samples: take first, ~1/4, ~1/2, ~3/4 to vary scenes.
    n = len(cand)
    idx = [0, n // 4, n // 2, max(0, n - 1)] if n >= n_panels else list(range(n))
    picks = [cand[i] for i in idx[:n_panels]]

    rows, cols = 2, 2
    fig, ax = plt.subplots(rows, cols, figsize=(11, 8))
    for k, p in enumerate(picks):
        r, c = divmod(k, cols)
        _imshow_with_title(ax[r, c], p, os.path.basename(p))
    for k in range(len(picks), rows * cols):
        r, c = divmod(k, cols)
        ax[r, c].axis("off")
    fig.suptitle(
        "Multi-modality — M-ATTN K=5 samples on inD/uniD val. "
        "Blue: obs (10 fr). Green: GT future. Red: K=5 sampled futures.",
        fontsize=12, y=0.995,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "qual_multimodal.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="tmp/figs")
    p.add_argument("--viz-dir", default="tmp/figs/viz_mattn_k5",
                   help="Dir holding per-agent PNGs from contextvae.visualize")
    p.add_argument("--only", default=None,
                   choices=["visdrone", "failure", "multimodal"])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.only in (None, "visdrone"):
        compose_visdrone(args.out_dir)
    if args.only in (None, "failure"):
        compose_failure_egomotion(args.out_dir)
    if args.only in (None, "multimodal"):
        compose_multimodal(args.viz_dir, args.out_dir)


if __name__ == "__main__":
    main()
