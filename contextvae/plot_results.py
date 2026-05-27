"""Render thesis Chapter 6 result figures from the locked numbers.

This script bakes in the K-sweep table from ``tmp/sweep_k.log`` (run on
2026-05-18) and the LSTM headline numbers from ``tmp/levelx_lstm.eval.log``
(run on 2026-05-27) so the figures can be regenerated deterministically
without re-running eval.

Run inside ``new_xupei_env``::

    python -m contextvae.plot_results --out-dir tmp/figs

Outputs (PNG, 200 dpi):

- ``k_sweep_ade.png`` / ``k_sweep_fde.png`` (A2)
- ``headline_ade_fde.png``                  (A3)
- ``mattn_relative_gain.png``               (A4)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K_STOCHASTIC = [1, 5, 10, 20]
MATTN = {
    "det": (0.5087, 1.4659),
    1:  (0.6986, 1.9893),
    5:  (0.3178, 0.7989),
    10: (0.2404, 0.5560),
    20: (0.1872, 0.3919),
}
SATTN = {
    "det": (0.5700, 1.6787),
    1:  (0.7834, 2.2710),
    5:  (0.3608, 0.9339),
    10: (0.2734, 0.6586),
    20: (0.2127, 0.4707),
}
LSTM_DET = (2.063, 4.103)


def plot_k_sweep(out_dir):
    for idx, label in enumerate(["minADE (m)", "minFDE (m)"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        m_y = [MATTN[k][idx] for k in K_STOCHASTIC]
        s_y = [SATTN[k][idx] for k in K_STOCHASTIC]
        ax.plot(K_STOCHASTIC, m_y, color="C0", marker="o",
                label="M-ATTN (orthomap)")
        ax.plot(K_STOCHASTIC, s_y, color="C1", marker="s",
                label="S-ATTN (no map)")
        ax.axhline(MATTN["det"][idx], color="C0", linestyle="--", alpha=0.7,
                   label=f"M-ATTN deterministic ({MATTN['det'][idx]:.3f})")
        ax.axhline(SATTN["det"][idx], color="C1", linestyle="--", alpha=0.7,
                   label=f"S-ATTN deterministic ({SATTN['det'][idx]:.3f})")
        ax.set_xscale("log")
        ax.set_xticks(K_STOCHASTIC)
        ax.set_xticklabels([str(k) for k in K_STOCHASTIC])
        ax.set_xlabel("K (number of sampled trajectories)")
        ax.set_ylabel(label)
        ax.set_title(f"{label.split(' ')[0]} vs K — inD+uniD+rounD val (520k samples)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        fname = "k_sweep_ade.png" if idx == 0 else "k_sweep_fde.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_headline(out_dir):
    models = ["LSTM (det)", "no-map S-ATTN\nK=20", "M-ATTN (orthomap)\nK=20"]
    ade_vals = [LSTM_DET[0], SATTN[20][0], MATTN[20][0]]
    fde_vals = [LSTM_DET[1], SATTN[20][1], MATTN[20][1]]
    colors = ["C3", "C1", "C0"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(models))
    for ax, vals, ylabel in zip(axes, [ade_vals, fde_vals],
                                ["minADE (m)", "minFDE (m)"]):
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" ")[0])
        ax.grid(alpha=0.3, axis="y", which="both")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, v * 1.05,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Headline comparison — inD+uniD+rounD val (lower is better)",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "headline_ade_fde.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_relative_gain(out_dir):
    ks = ["det", 1, 5, 10, 20]
    labels = ["det", "K=1", "K=5", "K=10", "K=20"]
    ade_gain = [100.0 * (SATTN[k][0] - MATTN[k][0]) / SATTN[k][0] for k in ks]
    fde_gain = [100.0 * (SATTN[k][1] - MATTN[k][1]) / SATTN[k][1] for k in ks]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(ks))
    w = 0.38
    b1 = ax.bar(x - w / 2, ade_gain, w, color="C0", label="minADE reduction",
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, fde_gain, w, color="C2", label="minFDE reduction",
                edgecolor="black", linewidth=0.5)
    for bars, vals in [(b1, ade_gain), (b2, fde_gain)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% reduction (M-ATTN vs no-map S-ATTN)")
    ax.set_title("M-ATTN marginal value across K — inD+uniD+rounD val")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "mattn_relative_gain.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default="tmp/figs")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_k_sweep(args.out_dir)
    plot_headline(args.out_dir)
    plot_relative_gain(args.out_dir)


if __name__ == "__main__":
    main()
