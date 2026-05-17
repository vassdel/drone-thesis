"""
Verify ContextVAEInferencer against ground truth on a real val sample.

Approach
--------
We bypass the training-side `contextvae.data.Dataloader` entirely and read
the raw `.txt` trajectory file directly. That gives us:
  - un-localized world-frame agent positions (xCenter, yCenter from levelXdata
    after the preprocessing notebook's 25 Hz -> 5 Hz frameskip + heading
    deg->rad conversion);
  - per-frame neighbor positions (all other agents at the same frame ids).

For one chosen target with a complete (OB_HORIZON + PRED_HORIZON) window we:
  1. Build target_history (OB_HORIZON x 2) and ground_truth_future (PRED_HORIZON x 2).
  2. Build neighbor_histories: Dict[track_id, (OB_HORIZON, 2)].
  3. Run ContextVAEInferencer.infer(target_history, neighbor_histories).
  4. Truncate ground_truth_future to PROTOCOL_FUTURE_STEPS (=6) and compute
     ADE/FDE against the inferencer's prediction.

A single-sample ADE in the same ballpark as the val-split-wide ADE=0.318 m
(from tmp/levelx_full_map_v1/metrics.png at epoch 57) is the bar we aim
for. We deliberately pick a "well-behaved" sample (>=N obs+pred frames, no
gaps) so this isn't a worst-case test.

Caveats
-------
- This is mean-over-K vs ground truth. The offline-eval ADE=0.318 is
  min-over-K — so deterministic/mean predictions can be a bit worse and
  still indicate the pipeline is correct. We tolerate up to ~2x the
  headline ADE for a single sample before complaining.
- The .txt schema written by `preprocessing/process_levelx.ipynb` is
      <frame_id> <agent_id> <x> <y> <heading_rad> <group_str>
  where group_str is one of "TARGET", "VEHICLE", "VRU" (per the CLAUDE.md
  notes). We filter neighbors against the same `inclusive_groups=["TARGET"]`
  contract the training-time loader uses by default.

How to run
----------
1. Activate the ContextVAE conda env (CUDA PyTorch build):
       conda activate new_xupei_env

2. Run as a plain script. `_REPO_ROOT` is derived from `__file__`, so the
   cwd does NOT matter — invoke from anywhere:
       python uav_guidance/server_code_only/main_program/verify_inference.py

   No CLI args; all paths are baked into `main()` and resolve relative to
   the repo root.

Required files (all paths relative to the ContextVAE repo root):
    data/levelx/val/inD_01.txt              # preprocessed val recording
    data/levelx/map/inD_01.pkl              # orthomap + homography
    configs/levelx_train.py                 # model config
    tmp/levelx_full_map_v1/ckpt-best        # M-ATTN headline checkpoint
                                            # (epoch 57, ADE 0.318 / FDE 0.800)

Hardware: GPU optional. The inferencer auto-selects CUDA if available
(A40 is the dev box); CPU works too — inference on one sample is ~seconds.

Expected output: a single ADE/FDE pair printed at the end. Headline (val-
split-wide, min-of-K=5) is ADE=0.318 m / FDE=0.800 m. This script computes
mean-of-K vs. ground truth on ONE sample, so 2-3x worse is normal. The
script only WARNs (does not exit non-zero) at >5x the headline — a real
pipeline bug should surface as either an exception or a clearly outsized
ADE/FDE.

Exit code: not set (script always returns 0). For an exit-coded wiring
regression on the swap, use `tests/test_contextvae_swap.py` instead. For
a visual overlay of the same prediction, use `visual_sanity.py` (writes
a PNG to tmp/).
"""

import os
import sys
import numpy as np

# Drop into the inferencer module's directory so its sys.path-shim runs.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from contextvae_inference import (  # noqa: E402
    ContextVAEInferencer,
    OB_HORIZON,
    PRED_HORIZON,
    PROTOCOL_FUTURE_STEPS,
    _REPO_ROOT,
)


# How many 25 Hz frames make up one 5 Hz step. Matches the frameskip baked
# in by preprocessing/process_levelx.ipynb. The .txt file already contains
# only the kept frames (every 5th), but agent_id <-> frame_id gaps still
# exist when an agent enters/leaves the scene mid-recording, so this is
# the step we use when looking up "the next 5 Hz frame".
FRAME_STEP = 1  # the .txt rows are ALREADY downsampled in the preprocessing
                # notebook; consecutive rows for the same agent are 5 Hz apart
                # in time. Set to 1 to step row-by-row in the file.


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
    (target_aid, frame_ids, target_positions_full, frames_to_rows)
        target_aid               : int, chosen target agent id
        frame_ids                : list of (ob_h+pred_h) consecutive fids
        target_positions_full    : (ob_h+pred_h, 2) ndarray, target's (x, y)
        frames_to_rows           : Dict[fid -> list of (aid, x, y, group)]
                                   used to fetch neighbors per frame
    """
    # Index: per-agent list of (fid, x, y, group) sorted by fid.
    per_agent: dict = {}
    for fid, aid, x, y, _heading, group in rows:
        per_agent.setdefault(aid, []).append((fid, x, y, group))
    for aid in per_agent:
        per_agent[aid].sort(key=lambda r: r[0])

    # Frame -> [(aid, x, y, group), ...]. We'll need this to look up neighbors
    # at any frame id.
    frames_to_rows: dict = {}
    for fid, aid, x, y, _h, group in rows:
        frames_to_rows.setdefault(fid, []).append((aid, x, y, group))

    # Find the first TARGET agent with a long-enough contiguous-in-file run.
    # The .txt's group field is a slash-separated multi-tag string like
    # "VEHICLE/TARGET" or "VRU". The training-time loader splits on "/" and
    # treats it as a list of tags (contextvae/data.py:613). So an agent is
    # ego-eligible iff "TARGET" appears in its tag list — i.e. iff the
    # group string contains "TARGET" as a token.
    #
    # "Contiguous in file" means: each consecutive pair of (fid, fid_next)
    # for this agent has the same delta. The preprocessing notebook
    # downsamples 25 Hz -> 5 Hz, so consecutive rows for the same agent
    # have fid_step = 1 (frames are renumbered in the .txt). We demand the
    # FIRST `ob_h + pred_h` entries have a uniform delta among themselves.
    def _has_target_tag(group_str: str) -> bool:
        return "TARGET" in group_str.split("/")

    for aid, run in per_agent.items():
        if len(run) < ob_h + pred_h:
            continue
        window = run[: ob_h + pred_h]
        # All rows in the ego window must carry the TARGET tag — otherwise
        # the agent transitions classes mid-window, which is rare and
        # disqualifying.
        if not all(_has_target_tag(g) for (_f, _x, _y, g) in window):
            continue
        # Uniform fid step across the window.
        fids = [r[0] for r in window]
        diffs = np.diff(fids)
        if len(diffs) == 0 or not (diffs == diffs[0]).all():
            continue
        # Found a clean window. Pack it.
        xys = np.array([(r[1], r[2]) for r in window], dtype=np.float64)
        return aid, fids, xys, frames_to_rows

    raise RuntimeError(
        f"No TARGET agent found with a clean (ob_h={ob_h} + pred_h={pred_h}) window."
    )


def _collect_neighbors(frame_ids, target_aid, frames_to_rows, ob_h=OB_HORIZON):
    """
    Build the neighbor-histories dict expected by the inferencer.

    Neighbors here = all OTHER agents (any group) that have a valid
    position at EVERY one of the FIRST ob_h frame ids. We use the same
    strict-presence rule that training does (no per-agent padding within
    a single window — agents missing at any obs frame are dropped).

    The trainer also filters by `inclusive_groups=["TARGET"]` (per
    configs/levelx_train.py:23). That filter applies to the EGO ROLE — the
    set of candidate target agents — NOT to neighbors. Neighbors include
    every other agent (VEHICLE/TARGET and VRU) within the observation
    radius. We mirror that here by NOT filtering neighbor groups.
    """
    obs_fids = frame_ids[:ob_h]
    # First pass: which neighbor aids are present at every obs frame.
    candidate: dict = {}  # aid -> list of (fid, x, y)
    for fid in obs_fids:
        present = {aid for (aid, _x, _y, _g) in frames_to_rows.get(fid, [])}
        if not candidate:
            # Initialize from the first frame.
            for aid, x, y, _g in frames_to_rows.get(fid, []):
                if aid == target_aid:
                    continue
                candidate[aid] = [(fid, x, y)]
        else:
            for aid in list(candidate.keys()):
                if aid not in present:
                    # Missed a frame; drop the candidate. The training
                    # loader's strict "present at every obs frame" rule.
                    del candidate[aid]
            # And add positions for the surviving candidates.
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

    target_aid, frame_ids, target_xys_full, frames_to_rows = _find_target_window(rows)
    print(f"[verify] picked target_aid={target_aid}")
    print(f"[verify] window fids: {frame_ids[0]}..{frame_ids[-1]} ({len(frame_ids)} frames)")
    target_history = target_xys_full[:OB_HORIZON]
    target_future_gt = target_xys_full[OB_HORIZON:]  # (PRED_HORIZON, 2)
    print(f"[verify] target history start: {target_history[0]}")
    print(f"[verify] target history end:   {target_history[-1]}")
    print(f"[verify] gt first future step: {target_future_gt[0]}")

    neighbors = _collect_neighbors(frame_ids, target_aid, frames_to_rows)
    print(f"[verify] found {len(neighbors)} neighbor(s) present at all obs frames")

    # Instantiate inferencer (pin map_model to resnet18 to match the
    # headline checkpoint, regardless of what configs/levelx_train.py
    # currently has).
    inf = ContextVAEInferencer(
        ckpt_path=ckpt,
        config_path=cfg,
        map_pickle_path=map_pickle,
        model_overrides={"map_model": "resnet18"},
    )

    pred = inf.infer(target_history, neighbors)
    print(f"[verify] pred shape: {pred.shape}")
    print(f"[verify] pred:\n{pred}")
    print(f"[verify] gt[:6]:\n{target_future_gt[:PROTOCOL_FUTURE_STEPS]}")

    # ADE / FDE over the 6-step horizon. Mean-over-K vs ground truth (not
    # min-over-K, so don't expect the val-split-wide 0.318 m headline).
    diffs = pred - target_future_gt[:PROTOCOL_FUTURE_STEPS]
    per_step_err = np.linalg.norm(diffs, axis=-1)
    ade = float(per_step_err.mean())
    fde = float(per_step_err[-1])
    print(f"[verify] ADE (6 steps, mean-of-K vs GT): {ade:.4f} m")
    print(f"[verify] FDE (6 steps, mean-of-K vs GT): {fde:.4f} m")

    # Single-sample tolerance. The val-wide eval is min-over-K=5; mean-of-K
    # on one sample can easily be 2-3x worse. We complain loudly if it's
    # more than 5x off, which would indicate a real pipeline bug rather
    # than sampling noise.
    HEADLINE_ADE = 0.3184
    HEADLINE_FDE = 0.7997
    if ade > 5.0 * HEADLINE_ADE:
        print(
            f"[verify] WARN: single-sample ADE {ade:.4f} >> 5x headline "
            f"{HEADLINE_ADE:.4f}; likely a pipeline issue."
        )
    if fde > 5.0 * HEADLINE_FDE:
        print(
            f"[verify] WARN: single-sample FDE {fde:.4f} >> 5x headline "
            f"{HEADLINE_FDE:.4f}; likely a pipeline issue."
        )
    print("[verify] done")


if __name__ == "__main__":
    main()
