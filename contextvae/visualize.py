"""Render per-agent inference visualizations: orthomap + observation history +
ground-truth future + K predicted samples, in agent-meter coordinates.

Mirrors `contextvae/main.py`'s config-loading, dataset construction, and
checkpoint-loading. Iterates the first --num_batches batches of the eval set,
runs `model(x, neighbor, *m, n_predictions=K)`, and writes one PNG per agent.

Usage (run from repo root with the project conda env active):

    conda activate new_xupei_env
    python -m contextvae.visualize \
        --config configs/levelx_eval.py \
        --ckpt   tmp/levelx_smoke/ckpt-best \
        --test   data/levelx/val \
        --map_dir data/levelx/map \
        --output tmp/levelx_smoke/viz \
        --num_batches 20 \
        --n_predictions 5

Notes:
  --ckpt accepts either a checkpoint file or a directory (resolved to
    <dir>/ckpt-best).
  --test takes one or more dataset roots (nargs='+').
  --test_map_dir defaults to --map_dir when omitted.
  --max_agents_per_batch caps PNGs per batch (default: all agents).
  --device defaults to cuda if available, else cpu.
"""
import os
import sys
import argparse
import importlib.util
import io

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contextvae.model import ContextVAE
from contextvae.data import Dataloader
from contextvae.utils import seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test", nargs='+', required=True)
    p.add_argument("--map_dir", type=str, default=None)
    p.add_argument("--test_map_dir", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--num_batches", type=int, default=20)
    p.add_argument("--n_predictions", type=int, default=5)
    p.add_argument("--max_agents_per_batch", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_config(path):
    spec = importlib.util.spec_from_file_location(
        "config", path,
        submodule_search_locations=[os.path.dirname(path)],
    )
    cfg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cfg
    spec.loader.exec_module(cfg)
    return cfg


def resolve_ckpt_path(ckpt_arg):
    if os.path.isfile(ckpt_arg):
        return ckpt_arg
    return os.path.join(ckpt_arg, "ckpt-best")


def get_recording_H(test_dataset, map_name):
    """Re-derive the per-recording homography H from the dataset's map dict.
    Mirrors data.py:347-356 to handle the four storage cases."""
    if not test_dataset.use_map:
        return None
    entry = test_dataset.map.get(map_name)
    if entry is None:
        return None
    if isinstance(entry, tuple):
        _, H = entry
        return H
    if isinstance(entry, str):
        _, H = Dataloader.load_map(entry, compressed=False)
        return H
    if isinstance(entry, io.BytesIO):
        entry.seek(0)
        loaded = np.load(entry)
        return loaded["H"]
    return None


def render_one(hist, future, preds, map_patch, m_per_px, seq_len, title, out_path):
    """Render a single (agent) PNG.

    hist:   (OB_HORIZON, 6) — col 0,1 are x,y in rotated frame; agent at hist[0].
    future: (PRED_HORIZON, 2)
    preds:  (K, PRED_HORIZON, 2)
    map_patch: (3, 224, 224) tensor in [-1, 1] or None.
    m_per_px:  per-recording meters/pixel (for extent), or None when no map.
    seq_len: int (valid history prefix) or None.
    """
    origin = hist[0, :2].copy()
    H_ego = hist[:, :2] - origin
    F_ego = future - origin
    P_ego = preds - origin

    if seq_len is not None and seq_len < H_ego.shape[0]:
        H_valid = H_ego[-seq_len:]
    else:
        H_valid = H_ego

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if map_patch is not None and m_per_px is not None:
        # 224x224 patch; agent sits at row 112, col 56 (asymmetric crop).
        x_min = -56 * m_per_px
        x_max = 168 * m_per_px
        y_min = -112 * m_per_px
        y_max = 112 * m_per_px
        img = map_patch.detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img + 1.0) * 0.5).clip(0.0, 1.0)
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max],
                  origin="upper", interpolation="bilinear", zorder=0)

    ax.plot(H_valid[:, 0], H_valid[:, 1], "-o",
            color="tab:blue", markersize=3, lw=1.5,
            label="Observation", zorder=3)
    ax.plot(F_ego[:, 0], F_ego[:, 1], "-o",
            color="tab:green", markersize=3, lw=1.5,
            label="GT future", zorder=3)
    K = P_ego.shape[0]
    for k in range(K):
        ax.plot(P_ego[k, :, 0], P_ego[k, :, 1], "-",
                color="tab:red", lw=1.0, alpha=0.5,
                label=f"Predicted (K={K})" if k == 0 else None,
                zorder=2)

    last_obs = H_valid[-1]
    ax.scatter([last_obs[0]], [last_obs[1]], c="black", marker="X",
               s=80, zorder=4, label="Last obs (t=0)")

    if map_patch is None:
        all_pts = np.concatenate(
            [H_valid, F_ego, P_ego.reshape(-1, 2)], axis=0)
        pad = 5.0
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m] (agent forward)")
    ax.set_ylabel("y [m] (agent left)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    if args.test_map_dir is None:
        args.test_map_dir = args.map_dir

    config = load_config(args.config)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device = torch.device(device)

    seed(args.seed)

    kwargs = dict(
        batch_first=False,
        device="cpu" if config.preload_data else device,
        seed=args.seed,
    )
    test_dataset = Dataloader(
        args.test, **kwargs,
        **config.test_dataloader,
        map_dir=args.test_map_dir,
        shuffle=False,
    )

    model = ContextVAE(**config.model)
    model.to(device)
    ckpt_path = resolve_ckpt_path(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    print(f"Load from ckpt: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    model.eval()

    if model.use_map != test_dataset.use_map:
        print(f"WARNING: model.use_map={model.use_map} but "
              f"test_dataset.use_map={test_dataset.use_map}. "
              f"This will likely fail at the forward call.")

    os.makedirs(args.output, exist_ok=True)

    # Materialize index lists deterministically (shuffle=False) so we can pair
    # each batch with the global dataset indices, hence with each agent's
    # recording id (and per-recording homography).
    sampler_iter = iter(test_dataset.batch_sampler)
    index_lists = []
    for _ in range(args.num_batches):
        try:
            index_lists.append(next(sampler_iter))
        except StopIteration:
            break

    saved = 0
    with torch.no_grad():
        for batch_idx, idx_list in enumerate(index_lists):
            items = [test_dataset[i] for i in idx_list]
            batch = test_dataset.collate_fn(items)
            if test_dataset.device != device:
                batch = [t.to(device) for t in batch]
            x, y, neighbor, *m = batch

            y_pred = model(x, neighbor, *m, n_predictions=args.n_predictions)

            map_tensor = None
            for t in m:
                if torch.is_tensor(t) and t.dim() == 5:
                    map_tensor = t
                    break
            seq_len_tensor = None
            for t in m:
                if torch.is_tensor(t) and t.dim() == 1 and t.dtype == torch.long:
                    seq_len_tensor = t
                    break

            N = x.size(1)
            n_to_save = N if args.max_agents_per_batch is None \
                else min(N, args.max_agents_per_batch)

            for n in range(n_to_save):
                global_idx = idx_list[n]
                map_name = test_dataset.data[global_idx][3][0]

                map_patch = map_tensor[0, n] if map_tensor is not None else None
                m_per_px = None
                if map_patch is not None:
                    H = get_recording_H(test_dataset, map_name)
                    if H is not None:
                        m_per_px = 1.0 / float(abs(H[1, 0]))

                seq_len_val = (
                    int(seq_len_tensor[n].item())
                    if seq_len_tensor is not None else None
                )

                rec_tag = map_name if map_name else "scene"
                fname = f"batch{batch_idx:03d}_agent{n:02d}_{rec_tag}.png"
                out_path = os.path.join(args.output, fname)
                title = f"{rec_tag}  b{batch_idx:03d}_a{n:02d}"

                render_one(
                    hist=x[:, n, :].detach().cpu().numpy(),
                    future=y[:, n, :].detach().cpu().numpy(),
                    preds=y_pred[:, :, n, :].detach().cpu().numpy(),
                    map_patch=map_patch,
                    m_per_px=m_per_px,
                    seq_len=seq_len_val,
                    title=title,
                    out_path=out_path,
                )
                saved += 1

            sys.stdout.write(
                f"\r\033[K Visualized batch {batch_idx + 1}/"
                f"{len(index_lists)} ({saved} PNGs saved)"
            )
            sys.stdout.flush()
    print()
    print(f"Wrote {saved} PNGs to {args.output}")


if __name__ == "__main__":
    main()
