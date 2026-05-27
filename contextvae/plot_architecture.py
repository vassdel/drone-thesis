"""Render the thesis Chapter 3/4 architecture diagrams.

All seven diagrams listed in the plan file are emitted as PNGs into ``--out-dir``
(default ``tmp/figs``).  Each diagram is a self-contained matplotlib block
drawing — no data is read from disk — so the figures regenerate deterministically.

Outputs:
- ``arch_c1_system.png``         End-to-end UAV → server → UAV (C1)
- ``arch_c2_contextvae.png``     ContextVAE forward pass (C2)
- ``arch_c3_mattn.png``          M-ATTN orthomap encoder (C3)
- ``arch_c4_sattn.png``          S-ATTN social attention block (C4)
- ``arch_c5_lstm_swap.png``      LSTM → ContextVAE Ntousis swap (C5)
- ``arch_c6_egomotion.png``      Ego-motion shim 3-stage chain (C6)
- ``arch_c7_timing.png``         Rolling-window timing (C7)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Consistent palette across all diagrams.
PALETTE = {
    "input":  "#cfe2f3",   # light blue
    "embed":  "#fce5cd",   # light orange
    "attn":   "#f4cccc",   # light red
    "rnn":    "#d9ead3",   # light green
    "latent": "#d9d2e9",   # light purple
    "output": "#e6e6e6",   # gray
    "ours":   "#fff2cc",   # light yellow (highlight contribution)
    "removed": "#f9d6d5",  # pinkish (struck-through)
}
EDGE = "#333333"


def _box(ax, x, y, w, h, text, color="#cfe2f3", fontsize=9, italic=False,
         strike=False, edge=EDGE, lw=1.0):
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, edgecolor=edge, facecolor=color)
    ax.add_patch(bb)
    style = {"fontsize": fontsize, "ha": "center", "va": "center"}
    if italic:
        style["style"] = "italic"
    txt = ax.text(x + w / 2, y + h / 2, text, **style)
    if strike:
        from matplotlib import patheffects as pe
        txt.set_path_effects([pe.withStroke(linewidth=0)])
        # Draw a strike line through the text centre.
        ax.plot([x + 0.1, x + w - 0.1], [y + h / 2, y + h / 2],
                color="#cc0000", lw=1.0, alpha=0.7)
    return (x, y, w, h)


def _arrow(ax, start, end, label=None, dashed=False, color=EDGE, label_offset=(0.0, 0.15),
           label_fontsize=8, curve=0.0):
    style = "->,head_length=8,head_width=5"
    linestyle = "--" if dashed else "-"
    arr = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                          linewidth=1.0, linestyle=linestyle,
                          connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=label_fontsize,
                style="italic", color="#555555")


def _setup(figsize, xlim, ylim, title=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    return fig, ax


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------- C1: system
def plot_c1(out_dir):
    fig, ax = _setup((13, 5.5), (0, 13), (0, 6),
                     title="End-to-end UAV trajectory-prediction system "
                           "(thesis adaptation of Ntousis 2024)")

    # Onboard track (top row).
    _box(ax, 0.2, 4.5, 1.8, 0.9, "UAV\ncamera", PALETTE["input"])
    _box(ax, 2.2, 4.5, 2.4, 0.9, "RPi 4B + NCS2\nYOLOv5n + KCF\n~20 FPS",
         PALETTE["embed"])
    _arrow(ax, (2.0, 4.95), (2.2, 4.95))
    _arrow(ax, (4.6, 4.95), (5.4, 4.95), label="bbox + crop", label_fontsize=8)

    # Network up.
    _box(ax, 5.4, 4.5, 1.6, 0.9, "Network\nuplink",
         PALETTE["output"], italic=True)
    _arrow(ax, (7.0, 4.95), (7.6, 4.95))

    # Server side.
    _box(ax, 7.6, 4.5, 2.4, 0.9, "YOLOv8x + DeepSORT\n(A40, ~15 FPS)",
         PALETTE["embed"])

    # Down from server detection to coord+model row.
    _arrow(ax, (8.8, 4.5), (8.8, 3.7), label="track centroids (px)",
           label_offset=(1.4, 0.0))

    # Coord shim row.
    _box(ax, 6.5, 2.8, 3.0, 0.9,
         "PixelMetricShim  |  MovingCameraShim\n(ORB+MAGSAC, 3-stage H)",
         PALETTE["ours"])
    _arrow(ax, (8.0, 2.8), (8.0, 2.0), label="(x,y) metric, world frame",
           label_offset=(1.5, 0.0))

    # ContextVAE inferencer.
    _box(ax, 5.5, 1.1, 5.0, 0.9,
         "ContextVAEInferencer  —  K=20 sampling, ResNet-18 M-ATTN\n"
         "input: x (10 obs), neighbors (radius 30m), heading-rotated orthomap",
         PALETTE["ours"])

    # Down to wire emission.
    _arrow(ax, (8.0, 1.1), (8.0, 0.55), label="6 future waypoints",
           label_offset=(1.0, 0.0))

    # Network down.
    _box(ax, 6.5, 0.0, 3.0, 0.55, "downlink → RPi guidance loop",
         PALETTE["output"], italic=True)

    # Loop back up to UAV.
    _arrow(ax, (6.5, 0.27), (1.1, 0.27), curve=-0.15,
           label="control set-point", label_offset=(0.0, -0.3))
    _arrow(ax, (1.1, 0.55), (1.1, 4.5))

    # Code-path annotations.
    ax.text(0.1, 5.7,
            "Onboard: server_code_only/main_program/...",
            fontsize=7, style="italic", color="#888888")
    ax.text(5.5, 2.05,
            "Files: scene_recog_socket_yolov8.py, contextvae_inference.py,\n"
            "pixel_metric_shim.py, ego_motion_shim.py",
            fontsize=7, style="italic", color="#555555")

    # Legend.
    legend = [
        mpatches.Patch(color=PALETTE["input"], label="data source"),
        mpatches.Patch(color=PALETTE["embed"], label="detection / tracking"),
        mpatches.Patch(color=PALETTE["ours"], label="this thesis (new code)"),
        mpatches.Patch(color=PALETTE["output"], label="I/O / network"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, frameon=False)

    _save(fig, os.path.join(out_dir, "arch_c1_system.png"))


# ---------------------------------------------------------------- C2: model
def plot_c2(out_dir):
    fig, ax = _setup((13, 8), (0, 13), (0, 8),
                     title="ContextVAE forward pass — timewise latent $z_t$, "
                           "S-ATTN + M-ATTN, training (solid) vs inference (dashed P/Q)")

    # ---- Inputs row.
    _box(ax, 0.3, 6.6, 1.6, 0.8, "$x$ : (L1+1, N, 6)\nobs state",
         PALETTE["input"])
    _box(ax, 2.1, 6.6, 2.4, 0.8,
         "neighbor : (L1+L2+1, N, $N_n$, 6)\nabsent padded with 1e9",
         PALETTE["input"])
    _box(ax, 4.7, 6.6, 1.8, 0.8,
         "map : (3, 224, 224)\nheading-rotated patch",
         PALETTE["input"])
    _box(ax, 6.7, 6.6, 1.8, 0.8,
         "$y$ : (L2, N, 2)\n(training only)",
         PALETTE["input"])

    # ---- Embedding layer.
    _box(ax, 0.3, 5.4, 1.6, 0.7, "embed_s\n→ 128", PALETTE["embed"])
    _box(ax, 2.1, 5.4, 1.4, 0.7, "embed_n\n→ 128", PALETTE["embed"])
    _box(ax, 3.6, 5.4, 1.6, 0.7, "embed_k\n→ 256 (keys)", PALETTE["embed"])
    _box(ax, 5.4, 5.4, 2.0, 0.7,
         "map_encode (ResNet-18)\n→ 512",
         PALETTE["ours"])

    _arrow(ax, (1.1, 6.6), (1.1, 6.1))
    _arrow(ax, (3.3, 6.6), (2.8, 6.1))
    _arrow(ax, (3.3, 6.6), (4.4, 6.1))
    _arrow(ax, (5.6, 6.6), (6.4, 6.1))

    # ---- S-ATTN box.
    _box(ax, 2.6, 4.1, 2.6, 0.8,
         "S-ATTN\nsoftmax($q^\\top k$, mask 1e9→−∞)",
         PALETTE["attn"])
    _box(ax, 5.4, 4.1, 2.0, 0.8,
         "M-ATTN init\n(map-conditioned)",
         PALETTE["attn"])
    _arrow(ax, (2.8, 5.4), (3.0, 4.9))
    _arrow(ax, (4.4, 5.4), (4.4, 4.9))
    _arrow(ax, (6.4, 5.4), (6.4, 4.9))

    # ---- Encoder RNN (forward over L1).
    _box(ax, 2.6, 2.8, 4.8, 0.9,
         "rnn_fx (forward GRU over L1=10 obs frames)\n"
         "hidden $h_t$ ∈ ℝ^{512}",
         PALETTE["rnn"])
    _arrow(ax, (3.9, 4.1), (3.9, 3.7))
    _arrow(ax, (1.1, 5.4), (3.0, 3.7), curve=-0.2)
    _arrow(ax, (6.4, 4.1), (5.8, 3.7))

    # ---- Backward encoder (future, training only).
    _box(ax, 7.7, 5.0, 3.2, 0.9,
         "rnn_by (backward GRU over L2=25)\n"
         "$b_t$ ∈ ℝ^{256}  — training only",
         PALETTE["rnn"], lw=1.4)
    _arrow(ax, (7.6, 7.0), (9.3, 5.9), dashed=True, label="future neighbors\n+ ego-state",
           label_offset=(1.4, 0.4))

    # ---- Per-t decoder loop box.
    _box(ax, 9.4, 0.6, 3.3, 3.6,
         "",
         PALETTE["latent"], lw=1.6)
    ax.text(11.05, 4.0, "Per-$t$ decoder loop  ($t = 1..H$)",
            ha="center", fontsize=10, style="italic")

    _box(ax, 9.6, 3.2, 1.4, 0.6, "$P_Z(h_t)$\nprior", PALETTE["latent"])
    _box(ax, 11.1, 3.2, 1.4, 0.6, "$Q_Z(h_t, b_t)$\nposterior", PALETTE["latent"])
    _box(ax, 10.05, 2.3, 2.0, 0.6, "$z_t$ ~ Normal\n(reparam.)", PALETTE["latent"])
    _box(ax, 10.05, 1.4, 2.0, 0.6, "DecoderZH\n$z_t, h_t → Δd_t$", PALETTE["rnn"])
    _box(ax, 10.05, 0.7, 2.0, 0.55,
         "EmbedZD ⊙\nrnn_fy step", PALETTE["rnn"])

    _arrow(ax, (10.3, 3.2), (10.7, 2.9))
    _arrow(ax, (11.8, 3.2), (11.4, 2.9), dashed=True,
           label="(training only)", label_offset=(0.3, 0.2))
    _arrow(ax, (11.05, 2.3), (11.05, 2.0))
    _arrow(ax, (11.05, 1.4), (11.05, 1.25))

    # h enters the loop.
    _arrow(ax, (7.4, 3.2), (9.6, 3.5), label="$h$", label_offset=(-1.0, 0.0))
    # b enters the loop (training only).
    _arrow(ax, (9.0, 5.0), (11.7, 3.8), dashed=True)
    # Recurrent feedback inside loop.
    _arrow(ax, (12.4, 0.95), (12.7, 2.0), curve=0.35)
    _arrow(ax, (12.7, 2.0), (10.3, 3.4), curve=0.35,
           label="$h_{t+1}$", label_offset=(-0.4, 0.4))

    # ---- Output.
    _box(ax, 4.0, 0.6, 4.0, 0.7,
         "$\\hat{y}_{1..H}$ — predicted trajectory  (H=25 = 5 s @ 5 Hz)",
         PALETTE["output"])
    _arrow(ax, (9.4, 0.95), (8.0, 0.95))

    # KL annotation.
    ax.text(11.05, 4.55,
            "$\\mathcal{L} = \\sum_t \\|\\hat{y}_t-y_t\\|_2^2 + \\beta\\,KL(Q_Z\\|P_Z)$",
            ha="center", fontsize=8, color="#555555")

    # Legend.
    legend = [
        mpatches.Patch(color=PALETTE["input"], label="tensor input"),
        mpatches.Patch(color=PALETTE["embed"], label="MLP embedding"),
        mpatches.Patch(color=PALETTE["attn"], label="attention"),
        mpatches.Patch(color=PALETTE["rnn"], label="GRU"),
        mpatches.Patch(color=PALETTE["latent"], label="latent / decoder"),
        mpatches.Patch(color=PALETTE["ours"], label="our M-ATTN contribution"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8, frameon=False,
              ncol=3)

    _save(fig, os.path.join(out_dir, "arch_c2_contextvae.png"))


# ---------------------------------------------------------------- C3: M-ATTN
def plot_c3(out_dir):
    fig, ax = _setup((12, 4.5), (0, 12), (0, 4.5),
                     title="M-ATTN — aerial-orthomap encoder (Chapter 3 contribution)")

    # Left: contrast with paper's HD raster.
    _box(ax, 0.2, 3.0, 2.6, 1.2,
         "Paper (Xu 2023)\nHD semantic raster\n(lanes, road edges)",
         PALETTE["removed"])
    _box(ax, 0.2, 1.0, 2.6, 1.2,
         "Ours\nAerial orthomap patch\nfrom $XX\\_background.png$",
         PALETTE["ours"])
    ax.text(1.5, 2.65, "vs", ha="center", fontsize=10, style="italic")

    # Center: forward path.
    _box(ax, 3.6, 1.8, 1.8, 0.9,
         "heading-rotated\ncrop ~30 m × 30 m",
         PALETTE["input"])
    _box(ax, 5.8, 1.8, 1.8, 0.9, "(3, 224, 224)\nin [-1, 1]", PALETTE["input"])
    _box(ax, 8.0, 1.8, 2.0, 0.9,
         "ResNet-18\nImageNet init,\nfreeze→fine-tune",
         PALETTE["embed"])
    _box(ax, 10.3, 1.8, 1.5, 0.9, "feature\nvec  (512,)", PALETTE["embed"])

    _arrow(ax, (5.4, 2.25), (5.8, 2.25))
    _arrow(ax, (7.6, 2.25), (8.0, 2.25))
    _arrow(ax, (10.0, 2.25), (10.3, 2.25))

    # Down to fusion.
    _arrow(ax, (11.0, 1.8), (11.0, 1.0))
    _box(ax, 7.5, 0.3, 4.3, 0.7,
         "fuse with attended-neighbor (256)  →  initial $h_0 \\in \\mathbb{R}^{512}$",
         PALETTE["rnn"])
    _arrow(ax, (9.5, 1.0), (9.5, 1.0 - 0.0))  # dummy

    # Code reference.
    ax.text(8.0, 3.55,
            "code: contextvae/context_vae.py — MapEncode class (lines 121–183)",
            fontsize=8, style="italic", color="#555555")

    _save(fig, os.path.join(out_dir, "arch_c3_mattn.png"))


# ---------------------------------------------------------------- C4: S-ATTN
def plot_c4(out_dir):
    fig, ax = _setup((11, 4.5), (0, 11), (0, 4.5),
                     title="S-ATTN — masked social attention over radius-r neighbors")

    _box(ax, 0.3, 3.4, 2.0, 0.8,
         "hidden $h_t$\n(N, 512)",
         PALETTE["input"])
    _box(ax, 0.3, 2.0, 2.0, 0.8,
         "neighbor features\n(N, $N_n$, 6)",
         PALETTE["input"])
    _box(ax, 0.3, 0.5, 2.0, 0.8,
         "distance mask\n$\\mathbf{1}[d \\leq 30\\,m]$",
         PALETTE["input"])

    _box(ax, 3.0, 3.4, 2.0, 0.8,
         "embed_q\n→ $q_t$ (N, 256)", PALETTE["embed"])
    _box(ax, 3.0, 2.0, 2.0, 0.8,
         "embed_k\n→ $k_t$ (N, $N_n$, 256)", PALETTE["embed"])

    _arrow(ax, (2.3, 3.8), (3.0, 3.8))
    _arrow(ax, (2.3, 2.4), (3.0, 2.4))

    # Score block.
    _box(ax, 5.6, 2.6, 1.8, 0.8,
         "$e_{ij} = k_{ij}^\\top q_i$",
         PALETTE["attn"])
    _box(ax, 5.6, 1.6, 1.8, 0.8,
         "LeakyReLU(0.2)", PALETTE["attn"])
    _box(ax, 5.6, 0.5, 1.8, 0.8,
         "$e_{ij} \\leftarrow -\\infty$\nif masked out",
         PALETTE["attn"])

    _arrow(ax, (5.0, 3.8), (6.0, 3.4))
    _arrow(ax, (5.0, 2.4), (6.0, 3.0))
    _arrow(ax, (2.3, 0.9), (5.6, 0.9))
    _arrow(ax, (6.5, 2.6), (6.5, 2.4))
    _arrow(ax, (6.5, 1.6), (6.5, 1.3))

    # Softmax and aggregate.
    _box(ax, 8.0, 1.6, 2.0, 0.8,
         "$\\alpha_i = \\mathrm{softmax}_j(e_{ij})$",
         PALETTE["attn"])
    _box(ax, 8.0, 0.5, 2.0, 0.8,
         "$\\bar n_i = \\sum_j \\alpha_{ij} n_{ij}$",
         PALETTE["rnn"])
    _arrow(ax, (7.4, 2.0), (8.0, 2.0))
    _arrow(ax, (9.0, 1.6), (9.0, 1.3))

    ax.text(5.5, 4.05,
            "1e9 padding sentinel + radius mask filter absent neighbors so they "
            "score $-\\infty$ → 0 weight",
            ha="center", fontsize=8, style="italic", color="#555555")

    _save(fig, os.path.join(out_dir, "arch_c4_sattn.png"))


# ---------------------------------------------------------------- C5: swap
def plot_c5(out_dir):
    fig, ax = _setup((13, 6.5), (0, 13), (0, 6.5),
                     title="From Ntousis (2024) LSTM predictor to ContextVAE — what changes")

    # ----------------- LEFT: Ntousis LSTM stack.
    ax.text(3.0, 6.1, "Ntousis (2024) — baseline", ha="center",
            fontsize=11, weight="bold")
    _box(ax, 0.5, 5.0, 5.0, 0.6,
         "DeepSORT tracks + 4-direction\ndetect_{up,down,left,right} neighbors",
         PALETTE["removed"], strike=True)
    _box(ax, 0.5, 4.0, 5.0, 0.6,
         "scale_data / unscale_data  (min-max → [0,1])",
         PALETTE["removed"], strike=True)
    _box(ax, 0.5, 3.0, 5.0, 0.6,
         "10 features × 25 history frames  (stateless per call)",
         PALETTE["removed"])
    _box(ax, 0.5, 2.0, 5.0, 0.6,
         "LSTM(input=10, hidden=64)",
         PALETTE["rnn"])
    _box(ax, 0.5, 1.0, 5.0, 0.6,
         "FC(64 → 50)  →  25 × (Δx, Δy)",
         PALETTE["embed"])
    _box(ax, 0.5, 0.1, 5.0, 0.55,
         "unimodal deterministic forecast",
         PALETTE["output"])

    for y_top, y_bot in [(5.0, 4.6), (4.0, 3.6), (3.0, 2.6), (2.0, 1.6),
                        (1.0, 0.65)]:
        _arrow(ax, (3.0, y_top), (3.0, y_bot))

    # ----------------- RIGHT: ContextVAE stack.
    ax.text(10.0, 6.1, "This thesis — ContextVAE adaptation", ha="center",
            fontsize=11, weight="bold")
    _box(ax, 7.5, 5.0, 5.0, 0.6,
         "radius-30m neighbor selection on DeepSORT centroids",
         PALETTE["ours"])
    _box(ax, 7.5, 4.0, 5.0, 0.6,
         "metric coords (PixelMetricShim / MovingCameraShim)",
         PALETTE["ours"])
    _box(ax, 7.5, 3.0, 5.0, 0.6,
         "x : (L1+1, N, 6),   neighbor : (L1+L2+1, N, $N_n$, 6),   map : (3,224,224)",
         PALETTE["input"])
    _box(ax, 7.5, 2.0, 5.0, 0.6,
         "ContextVAE  (S-ATTN + M-ATTN + timewise $z_t$)",
         PALETTE["latent"])
    _box(ax, 7.5, 1.0, 5.0, 0.6,
         "$K{=}20$ samples → mean (or top-K) → 6-waypoint protocol",
         PALETTE["attn"])
    _box(ax, 7.5, 0.1, 5.0, 0.55,
         "multi-modal stochastic forecast",
         PALETTE["output"])

    for y_top, y_bot in [(5.0, 4.6), (4.0, 3.6), (3.0, 2.6), (2.0, 1.6),
                        (1.0, 0.65)]:
        _arrow(ax, (10.0, y_top), (10.0, y_bot))

    # Center arrow / labels showing what's removed / added.
    ax.annotate("",
                xy=(7.3, 3.0), xytext=(5.7, 3.0),
                arrowprops=dict(arrowstyle="->", color="#777777", lw=1.4))
    ax.text(6.5, 3.25, "swap", ha="center", fontsize=10, style="italic",
            color="#555555")

    legend = [
        mpatches.Patch(color=PALETTE["removed"], label="removed in this thesis"),
        mpatches.Patch(color=PALETTE["ours"], label="new code"),
        mpatches.Patch(color=PALETTE["latent"], label="ContextVAE model"),
    ]
    ax.legend(handles=legend, loc="lower center", fontsize=8, ncol=3,
              frameon=False, bbox_to_anchor=(0.5, -0.04))

    _save(fig, os.path.join(out_dir, "arch_c5_lstm_swap.png"))


# ---------------------------------------------------------------- C6: ego-motion
def plot_c6(out_dir):
    fig, ax = _setup((13, 5.5), (0, 13), (0, 5.5),
                     title="MovingCameraShim — 3-stage homography chain "
                           "(live frame → keyframe Kc → keyframe K0 → world)")

    # Stage boxes.
    _box(ax, 0.2, 2.5, 2.4, 1.2,
         "Live frame\n$F_t$\n(BGR, 1920×1080)",
         PALETTE["input"])
    _box(ax, 3.5, 2.5, 3.0, 1.2,
         "Current keyframe $K_c$\n(re-anchored periodically)",
         PALETTE["embed"])
    _box(ax, 7.4, 2.5, 2.6, 1.2,
         "Reference keyframe $K_0$\n(anchored once at boot)",
         PALETTE["embed"])
    _box(ax, 10.8, 2.5, 2.0, 1.2,
         "World frame\n(metric, +y up)",
         PALETTE["output"])

    # Arrows with H labels.
    _arrow(ax, (2.6, 3.1), (3.5, 3.1),
           label="$H_{F_t \\to K_c}$\nORB+MAGSAC", label_offset=(0.0, 0.55),
           label_fontsize=8)
    _arrow(ax, (6.5, 3.1), (7.4, 3.1),
           label="$H_{K_c \\to K_0}$\ncached chain", label_offset=(0.0, 0.55),
           label_fontsize=8)
    _arrow(ax, (10.0, 3.1), (10.8, 3.1),
           label="$H_{K_0 \\to W}$\nfrom map.pkl", label_offset=(0.0, 0.55),
           label_fontsize=8)

    # Triggers and quality gates.
    _box(ax, 0.5, 0.5, 5.0, 1.3,
         "ORB+MAGSAC quality gates\n"
         "• inliers ≥ 50\n"
         "• inlier ratio ≥ 0.40 (else status = FROZEN)\n"
         "• reproj. threshold = 1.5 px\n"
         "• bbox interiors masked out of feature detection",
         PALETTE["attn"], fontsize=8.5)

    _box(ax, 6.0, 0.5, 6.7, 1.3,
         "Re-anchor trigger ($K_c \\leftarrow F_t$):\n"
         "• live↔Kc inlier ratio < 0.55\n"
         "• OR translation > 40 % image width\n"
         "Re-anchor refits $H_{K_c^{new} \\to K_0}$ "
         "against $K_0$'s permanent descriptors → no drift",
         PALETTE["latent"], fontsize=8.5)

    _arrow(ax, (3.0, 2.5), (3.0, 1.8))
    _arrow(ax, (9.0, 2.5), (9.0, 1.8))

    ax.text(6.5, 4.7,
            "Status returned per call: OK / REANCHORED / FROZEN. "
            "Active downstream of YOLO detection; same 4-method API as PixelMetricShim.",
            ha="center", fontsize=8, style="italic", color="#555555")
    ax.text(6.5, 0.2,
            "code: uav_guidance/server_code_only/main_program/ego_motion_shim.py",
            ha="center", fontsize=7.5, style="italic", color="#888888")

    _save(fig, os.path.join(out_dir, "arch_c6_egomotion.png"))


# ---------------------------------------------------------------- C7: timing
def plot_c7(out_dir):
    fig, ax = _setup((13, 5.5), (0, 13), (0, 5.5),
                     title="Rolling-window inference — detection at 20 FPS, "
                           "ContextVAE forecast at 5 Hz")

    # Timeline.
    t_min, t_max = 1.0, 12.0
    ax.plot([t_min, t_max], [3.0, 3.0], color="black", lw=1.0)
    for tx in [1, 2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8, 10.9]:
        ax.plot([tx], [3.0], marker="|", markersize=10, color="black")
    ax.text(t_max + 0.1, 3.0, "wall time →", va="center", fontsize=9)

    # Detection ticks (every 50 ms ~ 20 FPS).
    det_pts = [(1.0 + 0.275 * i, 3.45) for i in range(0, 40)]
    for x, y in det_pts:
        if x > t_max:
            continue
        ax.plot([x], [y], marker="v", color="#1f77b4", markersize=5)
    ax.text(t_max + 0.1, 3.45, "YOLO+DeepSORT (20 FPS, 50 ms)",
            va="center", fontsize=8, color="#1f77b4")

    # ContextVAE inference (every 200 ms ~ 5 Hz).
    inf_pts = [(1.0 + 1.1 * i, 2.55) for i in range(0, 11)]
    for x, y in inf_pts:
        if x > t_max:
            continue
        ax.plot([x], [y], marker="^", color="#d62728", markersize=8)
    ax.text(t_max + 0.1, 2.55,
            "ContextVAEInferencer (5 Hz, 200 ms)  — 27 ms median latency on A40",
            va="center", fontsize=8, color="#d62728")

    # Observation window (sliding).
    obs_x_start = 4.0
    obs_x_end = 6.2
    _box(ax, obs_x_start, 1.4, obs_x_end - obs_x_start, 0.6,
         "obs window  L1 = 10 frames @ 5 Hz  (2 s)",
         PALETTE["input"], fontsize=8)
    _arrow(ax, (obs_x_end, 1.7), (obs_x_end + 0.4, 1.7),
           label="slides 200 ms per inference", label_offset=(0.6, 0.4),
           label_fontsize=7)

    # Prediction horizon.
    pred_x_start = 6.4
    pred_x_end = 11.5
    _box(ax, pred_x_start, 0.6, pred_x_end - pred_x_start, 0.6,
         "prediction horizon  H = 25 frames (5 s) "
         "→ trimmed to PROTOCOL_FUTURE_STEPS=6 (1.2 s) wire-emitted",
         PALETTE["output"], fontsize=8)

    _arrow(ax, (6.3, 1.5), (6.3, 1.0))

    # Annotation: 5 Hz cadence enforced.
    ax.text(2.0, 0.2,
            "History pushed only every CONTEXTVAE_TARGET_DT_MS = 200 ms "
            "→ matches training cadence (no resampling mismatch).",
            fontsize=8, style="italic", color="#555555")

    ax.text(7.0, 4.6,
            "Detection runs faster than inference to keep tracker IDs warm; "
            "inference subsamples on the 5 Hz grid.",
            ha="center", fontsize=8, style="italic", color="#555555")

    _save(fig, os.path.join(out_dir, "arch_c7_timing.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="tmp/figs")
    p.add_argument("--only", default=None,
                   choices=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plots = {"c1": plot_c1, "c2": plot_c2, "c3": plot_c3, "c4": plot_c4,
             "c5": plot_c5, "c6": plot_c6, "c7": plot_c7}
    keys = [args.only] if args.only else list(plots.keys())
    for k in keys:
        plots[k](args.out_dir)


if __name__ == "__main__":
    main()
