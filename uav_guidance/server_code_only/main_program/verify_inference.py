"""
Verify ContextVAEInferencer against ground truth on a real val sample.

Approach
--------
Bypasses the training-side `contextvae.data.Dataloader` and reads the raw
`.txt` trajectory file directly, giving un-localized world-frame positions and
per-frame neighbor positions. For one TARGET agent with a complete
(OB_HORIZON + PRED_HORIZON) window it builds the target history, neighbor
histories, and ground-truth future, runs `infer()`, and reports ADE/FDE over
the first PROTOCOL_FUTURE_STEPS. A deliberately well-behaved sample (full
window, no gaps) is chosen — this is a correctness check, not a worst case.

Caveats
-------
- This compares mean-over-K against ground truth, whereas the headline
  ADE/FDE is min-over-K, so a single-sample mean prediction being a few times
  worse is normal. The script only warns (never exits non-zero) above 5x the
  headline — a real bug should surface as an exception or a grossly outsized
  error.
- The .txt schema is `<frame_id> <agent_id> <x> <y> <heading_rad> <group_str>`
  with group_str one of "TARGET"/"VEHICLE"/"VRU". The ego role is restricted
  to TARGET-tagged agents (the loader's `inclusive_groups=["TARGET"]`);
  neighbors are not group-filtered.

How to run
----------
Activate the ContextVAE conda env (CUDA PyTorch), then run as a plain script;
`_REPO_ROOT` is derived from `__file__`, so cwd does not matter:
    python uav_guidance/server_code_only/main_program/verify_inference.py

No CLI args. Required files (relative to the repo root):
    data/levelx/val/inD_01.txt              # preprocessed val recording
    data/levelx/map/inD_01.pkl              # orthomap + homography
    configs/levelx_train.py                 # model config
    tmp/levelx_full_map_v1/ckpt-best        # M-ATTN headline checkpoint

GPU optional (auto-selected if available); CPU runs one sample in seconds.
For an exit-coded wiring regression use `tests/test_contextvae_swap.py`; for a
visual overlay use `visual_sanity.py`.
"""

import os
import sys
import numpy as np

# Put the module dir on sys.path so sibling imports resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from contextvae_inference import (  # noqa: E402
    ContextVAEInferencer,
    OB_HORIZON,
    PRED_HORIZON,
    PROTOCOL_FUTURE_STEPS,
    _REPO_ROOT,
)


# The .txt rows are already downsampled to 5 Hz by the preprocessing notebook,
# so consecutive rows for an agent are one 5 Hz step apart; step row-by-row.
FRAME_STEP = 1


def _read_txt(path: str):
    """
    Read a preprocessed `.txt` file written by `preprocessing/process_levelx.ipynb`.

    Returns
    -------
    rows : list of (fid:int, aid:int, x:float, y:float, heading:float, group:str)
    """
    rows = []
    with open(path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            aid = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            heading = float(parts[4])
            group = parts[5]
            rows.append((fid, aid, x, y, heading, group))
    return rows


def _find_target_window(rows, ob_h=OB_HORIZON, pred_h=PRED_HORIZON):
    """
    Pick one TARGET agent that has at least (ob_h + pred_h) consecutive
    5 Hz frames in the file. Return enough metadata to build the test inputs.

    Returns
    -------
    (target_aid, frame_ids, target_positions_full, target_heading_world,
     frames_to_rows)
        target_aid               : int, chosen target agent id
        frame_ids                : list of (ob_h+pred_h) consecutive fids
        target_positions_full    : (ob_h+pred_h, 2) ndarray, target's (x, y)
        target_heading_world     : float, heading (radians, world frame) at
                                   the FIRST obs frame — column 5 of the .txt,
                                   the value the training loader uses
        frames_to_rows           : Dict[fid -> list of (aid, x, y, group)]
                                   used to fetch neighbors per frame
    """
    # Per-agent list of (fid, x, y, heading, group), sorted by fid. Heading is
    # kept so we can return the first-obs-frame value for the chosen target,
    # matching the loader and the inferencer's `heading_world` arg.
    per_agent: dict = {}
    for fid, aid, x, y, heading, group in rows:
        per_agent.setdefault(aid, []).append((fid, x, y, heading, group))
    for aid in per_agent:
        per_agent[aid].sort(key=lambda r: r[0])

    # Frame -> [(aid, x, y, group), ...] for neighbor lookup. Neighbor heading
    # is unused (only the target's heading drives localization/map rotation).
    frames_to_rows: dict = {}
    for fid, aid, x, y, _h, group in rows:
        frames_to_rows.setdefault(fid, []).append((aid, x, y, group))

    # Find the first TARGET agent with a long-enough contiguous run. The group
    # field is a slash-separated tag string (e.g. "VEHICLE/TARGET"); an agent
    # is ego-eligible iff "TARGET" is among its tags. "Contiguous" means the
    # first ob_h + pred_h rows have a uniform fid step (rows are already 5 Hz).
    def _has_target_tag(group_str: str) -> bool:
        return "TARGET" in group_str.split("/")

    for aid, run in per_agent.items():
        if len(run) < ob_h + pred_h:
            continue
        window = run[: ob_h + pred_h]
        # Every row in the window must carry the TARGET tag (reject agents that
        # change class mid-window).
        if not all(_has_target_tag(g) for (_f, _x, _y, _h, g) in window):
            continue
        # Uniform fid step across the window.
        fids = [r[0] for r in window]
        diffs = np.diff(fids)
        if len(diffs) == 0 or not (diffs == diffs[0]).all():
            continue
        # Clean window: return the positions and the first-obs-frame heading
        # (the value the loader uses for localization).
        xys = np.array([(r[1], r[2]) for r in window], dtype=np.float64)
        heading_world = float(window[0][3])  # already radians per preprocessing
        return aid, fids, xys, heading_world, frames_to_rows

    raise RuntimeError(
        f"No TARGET agent found with a clean (ob_h={ob_h} + pred_h={pred_h}) window."
    )


def _collect_neighbors(frame_ids, target_aid, frames_to_rows, ob_h=OB_HORIZON):
    """
    Build the neighbor-histories dict expected by the inferencer.

    Neighbors = all OTHER agents (any group) present at EVERY one of the first
    ob_h frames, matching the loader's strict presence rule (agents missing at
    any obs frame are dropped, no per-window padding).

    The loader's `inclusive_groups=["TARGET"]` filter applies to the EGO ROLE
    (candidate targets), not to neighbors, so neighbor groups are not filtered
    here.
    """
    obs_fids = frame_ids[:ob_h]
    # Track aids present at every obs frame.
    candidate: dict = {}  # aid -> list of (fid, x, y)
    for fid in obs_fids:
        present = {aid for (aid, _x, _y, _g) in frames_to_rows.get(fid, [])}
        if not candidate:
            # Seed from the first obs frame.
            for aid, x, y, _g in frames_to_rows.get(fid, []):
                if aid == target_aid:
                    continue
                candidate[aid] = [(fid, x, y)]
        else:
            for aid in list(candidate.keys()):
                if aid not in present:
                    # Dropped at this frame -> no longer a candidate.
                    del candidate[aid]
            # Extend surviving candidates with this frame's positions.
            for aid, x, y, _g in frames_to_rows.get(fid, []):
                if aid in candidate:
                    candidate[aid].append((fid, x, y))

    out = {}
    for aid, run in candidate.items():
        if len(run) != ob_h:
            continue
        out[aid] = np.array([(x, y) for (_f, x, y) in run], dtype=np.float64)
    return out


def main():
    val_txt = os.path.join(_REPO_ROOT, "data", "levelx", "val", "inD_01.txt")
    map_pickle = os.path.join(_REPO_ROOT, "data", "levelx", "map", "inD_01.pkl")
    ckpt = os.path.join(_REPO_ROOT, "tmp", "levelx_full_map_v1", "ckpt-best")
    cfg = os.path.join(_REPO_ROOT, "configs", "levelx_train.py")

    print(f"[verify] reading {val_txt}")
    rows = _read_txt(val_txt)
    print(f"[verify]   {len(rows)} rows")

    target_aid, frame_ids, target_xys_full, target_heading_world, frames_to_rows = (
        _find_target_window(rows)
    )
    print(f"[verify] picked target_aid={target_aid}")
    print(f"[verify] window fids: {frame_ids[0]}..{frame_ids[-1]} ({len(frame_ids)} frames)")
    target_history = target_xys_full[:OB_HORIZON]
    target_future_gt = target_xys_full[OB_HORIZON:]  # (PRED_HORIZON, 2)
    print(f"[verify] target history start: {target_history[0]}")
    print(f"[verify] target history end:   {target_history[-1]}")
    print(f"[verify] gt first future step: {target_future_gt[0]}")
    print(
        f"[verify] target heading at obs[0] (rad, world): "
        f"{target_heading_world:.6f}  (= {np.degrees(target_heading_world):.2f} deg)"
    )

    neighbors = _collect_neighbors(frame_ids, target_aid, frames_to_rows)
    print(f"[verify] found {len(neighbors)} neighbor(s) present at all obs frames")

    # Pin map_model to resnet18 to match the headline checkpoint regardless of
    # the config file's current contents.
    inf = ContextVAEInferencer(
        ckpt_path=ckpt,
        config_path=cfg,
        map_pickle_path=map_pickle,
        model_overrides={"map_model": "resnet18"},
    )

    gt6 = target_future_gt[:PROTOCOL_FUTURE_STEPS]

    # -----------------------------------------------------------------------
    # Run 1: heading_world=None forces the atan2(v0_y, v0_x) fallback inside
    # `infer()`. This diverges from the training loader (which reads the stored
    # heading), so it is the degraded baseline.
    # -----------------------------------------------------------------------
    print("\n[verify] === Run 1: heading_world=None (legacy atan2 fallback) ===")
    pred_fallback = inf.infer(target_history, neighbors, heading_world=None)
    diffs_fb = pred_fallback - gt6
    per_step_fb = np.linalg.norm(diffs_fb, axis=-1)
    ade_fb = float(per_step_fb.mean())
    fde_fb = float(per_step_fb[-1])
    print(f"[verify] BEFORE  ADE (fallback heading): {ade_fb:.4f} m")
    print(f"[verify] BEFORE  FDE (fallback heading): {fde_fb:.4f} m")

    # -----------------------------------------------------------------------
    # Run 2: pass the stored heading from the .txt (radians, world frame), as
    # the training loader uses it, so the map crop matches training.
    # -----------------------------------------------------------------------
    print("\n[verify] === Run 2: heading_world=actual heading (training-parity) ===")
    pred_real = inf.infer(
        target_history, neighbors, heading_world=target_heading_world
    )
    diffs_re = pred_real - gt6
    per_step_re = np.linalg.norm(diffs_re, axis=-1)
    ade_re = float(per_step_re.mean())
    fde_re = float(per_step_re[-1])
    print(f"[verify] AFTER   ADE (real heading):     {ade_re:.4f} m")
    print(f"[verify] AFTER   FDE (real heading):     {fde_re:.4f} m")

    # Side-by-side delta between the two heading sources.
    print("\n[verify] === Delta (AFTER - BEFORE) ===")
    print(f"[verify] dADE = {ade_re - ade_fb:+.4f} m  "
          f"({100.0 * (ade_re - ade_fb) / max(ade_fb, 1e-9):+.1f}%)")
    print(f"[verify] dFDE = {fde_re - fde_fb:+.4f} m  "
          f"({100.0 * (fde_re - fde_fb) / max(fde_fb, 1e-9):+.1f}%)")

    # Single-sample tolerance check against the headline, using the real-heading
    # (AFTER) number as the apples-to-apples comparison. Mean-of-K on one sample
    # can be 2-3x the val-wide min-of-K headline; warn only above 5x.
    HEADLINE_ADE = 0.3184
    HEADLINE_FDE = 0.7997
    if ade_re > 5.0 * HEADLINE_ADE:
        print(
            f"[verify] WARN: AFTER single-sample ADE {ade_re:.4f} >> 5x headline "
            f"{HEADLINE_ADE:.4f}; likely a pipeline issue."
        )
    if fde_re > 5.0 * HEADLINE_FDE:
        print(
            f"[verify] WARN: AFTER single-sample FDE {fde_re:.4f} >> 5x headline "
            f"{HEADLINE_FDE:.4f}; likely a pipeline issue."
        )
    print("[verify] done")


if __name__ == "__main__":
    main()
