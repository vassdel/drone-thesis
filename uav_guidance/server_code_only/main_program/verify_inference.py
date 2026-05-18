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
    (target_aid, frame_ids, target_positions_full, target_heading_world,
     frames_to_rows)
        target_aid               : int, chosen target agent id
        frame_ids                : list of (ob_h+pred_h) consecutive fids
        target_positions_full    : (ob_h+pred_h, 2) ndarray, target's (x, y)
        target_heading_world     : float, heading (radians, world frame) at
                                   the FIRST obs frame of the window — the
                                   value stored in column 5 of the .txt and
                                   used by the training loader (data.py:468)
        frames_to_rows           : Dict[fid -> list of (aid, x, y, group)]
                                   used to fetch neighbors per frame
    """
    # Index: per-agent list of (fid, x, y, heading, group) sorted by fid.
    # We MUST keep the heading per-row so we can return the value at the
    # first obs frame for the chosen target — the training-time loader
    # reads this exact value (contextvae/data.py:468), and the
    # ContextVAEInferencer needs it via the new `heading_world` arg to
    # match training-time behavior.
    per_agent: dict = {}
    for fid, aid, x, y, heading, group in rows:
        per_agent.setdefault(aid, []).append((fid, x, y, heading, group))
    for aid in per_agent:
        per_agent[aid].sort(key=lambda r: r[0])

    # Frame -> [(aid, x, y, group), ...]. We'll need this to look up neighbors
    # at any frame id. Heading is not needed for neighbors (the inferencer
    # only uses the target's heading for localization/map rotation).
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
        if not all(_has_target_tag(g) for (_f, _x, _y, _h, g) in window):
            continue
        # Uniform fid step across the window.
        fids = [r[0] for r in window]
        diffs = np.diff(fids)
        if len(diffs) == 0 or not (diffs == diffs[0]).all():
            continue
        # Found a clean window. Pack it. The heading at the FIRST obs frame
        # is what the training loader uses for localization (data.py:468 —
        # heading is captured at the first observation step of the window),
        # so we hand back exactly that scalar.
        xys = np.array([(r[1], r[2]) for r in window], dtype=np.float64)
        heading_world = float(window[0][3])  # already in radians per preprocessing
        return aid, fids, xys, heading_world, frames_to_rows

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

    # Instantiate inferencer (pin map_model to resnet18 to match the
    # headline checkpoint, regardless of what configs/levelx_train.py
    # currently has).
    inf = ContextVAEInferencer(
        ckpt_path=ckpt,
        config_path=cfg,
        map_pickle_path=map_pickle,
        model_overrides={"map_model": "resnet18"},
    )

    gt6 = target_future_gt[:PROTOCOL_FUTURE_STEPS]

    # -----------------------------------------------------------------------
    # Run 1: BEFORE the fix — heading_world=None forces the legacy
    # atan2(v0_y, v0_x) fallback inside `infer()`. This is the path the
    # "verified ADE 0.254 m" number was measured on. The fallback diverges
    # from training-time behavior (training reads the stored heading column),
    # so this number is the degraded baseline.
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
    # Run 2: AFTER the fix — pass the heading read from the .txt's column 5
    # (radians, world frame, exactly as the training loader uses it). The
    # resulting map crop matches the training-time distribution.
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

    # Side-by-side delta for the reveal.
    print("\n[verify] === Delta (AFTER - BEFORE) ===")
    print(f"[verify] dADE = {ade_re - ade_fb:+.4f} m  "
          f"({100.0 * (ade_re - ade_fb) / max(ade_fb, 1e-9):+.1f}%)")
    print(f"[verify] dFDE = {fde_re - fde_fb:+.4f} m  "
          f"({100.0 * (fde_re - fde_fb) / max(fde_fb, 1e-9):+.1f}%)")

    # Single-sample tolerance check against the headline (only the AFTER
    # number is the apples-to-apples comparison — the fix gets us back onto
    # the training-time distribution). Mean-of-K on one sample can easily
    # be 2-3x worse than the val-wide min-of-K=5 headline; we still complain
    # at >5x as a sanity floor.
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
