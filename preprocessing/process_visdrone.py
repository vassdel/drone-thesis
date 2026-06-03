#!/usr/bin/env python3
"""
process_visdrone.py — VisDrone2019-MOT  ->  ContextVAE on-disk .txt format.

Cross-domain evaluation preprocessor (thesis Week 4, Experiment 3). Converts an
official VisDrone-MOT split into the same `<name>.txt` rows that
`contextvae/data.py` consumes, so the EXISTING `contextvae/main.py` eval loop
(deterministic ADE/FDE + minADE_k/minFDE_k) can be run on VisDrone with no
changes — giving numbers computed identically to the inD/uniD/rounD val numbers.

Conversion is SCALE-ONLY (locked decision): a per-sequence scalar metres-per-pixel
calibrated from the known length of a typical sedan (~4.5 m), exactly as the
qualitative replay did. There is NO ego-motion compensation here, so on
moving-camera clips the metric is partly contaminated by drone motion. Report
every resulting number with the documented ±~30 % scale band — never as an
apples-to-apples figure against inD/uniD. (See `pixel_metric_shim.py:241-246`
and docs/w4_ablations_results.md §4.2.)

On-disk output (one file per sequence, under <out>/):
    <seq>.txt   rows:  fid aid x y heading group
                       - fid    : 5-Hz frame index, renumbered 0,1,2,...
                       - aid    : VisDrone target_id
                       - x, y   : metric world coords (metres), +y UP
                       - heading: radians, finite-diff atan2 of world motion
                       - group  : "VEHICLE/TARGET" (cars/vans/trucks/buses) or
                                  "VRU" (pedestrians/people/bikes/tricycles/motor)

Velocities/accelerations are NOT written: data.py's extend() derives them as
per-step position deltas (vx = x[t+1]-x[t]). We only have to guarantee a uniform
5-Hz frame grid in metric units, which this script produces.

Why scale-only needs no images for the geometry: bbox centres + a scalar m/px
are enough. We DO open one frame per sequence purely to read (W, H) for the
origin (frame centre). Origin choice is cosmetic — ADE/FDE are
translation/reflection invariant and the loader ego-normalises each window — so
the only thing that matters numerically is the m/px scale and the +y flip.

Usage
-----
    PY=/home/vdelis/anaconda3/envs/new_xupei_env/bin/python
    $PY preprocessing/process_visdrone.py \
        --root full-visdrone \
        --out  data/visdrone_levelx/val \
        --fps  30 --target-hz 5

The VisDrone-MOT layout expected under --root:
    <root>/annotations/<seq>.txt
    <root>/sequences/<seq>/<frame>.jpg
"""

import argparse
import math
import os
import sys
from collections import defaultdict


# VisDrone object_category codes (col 8 of the MOT annotation rows).
#   0 ignored  1 pedestrian  2 people  3 bicycle  4 car  5 van  6 truck
#   7 tricycle 8 awning-tricycle  9 bus  10 motor  11 others
# Vehicles match the inD/uniD TARGET classes the model was trained on
# (car/van/truck/bus). Everything else trackable is kept as VRU context so the
# S-ATTN neighbour set mirrors training (where VRUs are neighbours, never egos).
VEHICLE_CLASSES = frozenset({4, 5, 6, 9})
VRU_CLASSES = frozenset({1, 2, 3, 7, 8, 10})
# 0 (ignored regions) and 11 (others) are dropped entirely.

ASSUMED_CAR_LEN_M = 4.5  # typical sedan length; same heuristic as the replay run.


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="VisDrone-MOT split root (has annotations/ + sequences/).")
    p.add_argument("--out", required=True,
                   help="Output dir for <seq>.txt files (created if missing).")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Source frame rate of the sequences (VisDrone-MOT ~30 Hz).")
    p.add_argument("--target-hz", type=float, default=5.0,
                   help="Target rate to match ContextVAE training (5 Hz).")
    p.add_argument("--assumed-car-len", type=float, default=ASSUMED_CAR_LEN_M,
                   help="Assumed sedan length (m) for the m/px calibration.")
    p.add_argument("--default-mpp", type=float, default=0.045,
                   help="Fallback m/px when a sequence has no car-class tracks.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N sequences (smoke test).")
    return p.parse_args()


def read_frame_size(seq_dir):
    """Return (W, H) of the first frame in seq_dir, or None if unreadable."""
    try:
        frames = sorted(f for f in os.listdir(seq_dir)
                        if f.lower().endswith((".jpg", ".png", ".jpeg")))
        if not frames:
            return None
        from PIL import Image
        with Image.open(os.path.join(seq_dir, frames[0])) as im:
            return im.size  # (W, H)
    except Exception as e:
        print(f"  [warn] could not read frame size for {seq_dir}: {e}")
        return None


def load_mot_rows(anno_path):
    """
    Parse a VisDrone-MOT annotation file.

    Returns rows as a list of dicts and the set of distinct frame indices.
    Columns: frame, target_id, left, top, w, h, score, category, trunc, occ.
    """
    rows = []
    with open(anno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            left = float(parts[2])
            top = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            cat = int(float(parts[7]))
            rows.append(dict(frame=frame, tid=tid, left=left, top=top,
                             w=w, h=h, cat=cat))
    return rows


def _median_first_long_axis(rows, cat):
    """Median long-axis (max(w,h) px) over `cat`-class tracks at first appearance."""
    first_frame, first_long = {}, {}
    for r in rows:
        if r["cat"] != cat:
            continue
        tid = r["tid"]
        if tid not in first_frame or r["frame"] < first_frame[tid]:
            first_frame[tid] = r["frame"]
            first_long[tid] = max(r["w"], r["h"])
    if not first_long:
        return None, 0
    vals = sorted(first_long.values())
    n = len(vals)
    median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return (median if median > 0 else None), n


def calibrate_m_per_px(rows, assumed_car_len, default_mpp):
    """
    m/px from the median long-axis of a known-length vehicle class at first
    appearance. Mirrors the replay calibration that produced 0.04018 on
    uav0000305_00000_v. Fallback chain for car-less scenes (common across the
    larger MOT splits): cars (class 4, ~4.5 m) -> vans (class 5, ~4.8 m) ->
    hard default. Trucks/buses are NOT used — their lengths are too variable to
    calibrate from. Returns (m_per_px, n_tracks_used, source_label).
    """
    for cat, label, length in [(4, "car", assumed_car_len), (5, "van", 4.8)]:
        median, n = _median_first_long_axis(rows, cat)
        if median:
            return length / median, n, label
    return default_mpp, 0, "default"


def group_for(cat):
    if cat in VEHICLE_CLASSES:
        return "VEHICLE/TARGET"
    if cat in VRU_CLASSES:
        return "VRU"
    return None  # dropped (ignored / others)


def headings_from_positions(seq_xy):
    """
    Centred finite-difference atan2 heading (radians, +y up, CCW from +x) for an
    ordered list of (x, y) world positions. Endpoints use one-sided diffs.
    Stationary points fall back to the previous valid heading (else 0.0).
    """
    n = len(seq_xy)
    headings = [0.0] * n
    last = 0.0
    for k in range(n):
        if n == 1:
            dx = dy = 0.0
        elif k == 0:
            dx = seq_xy[1][0] - seq_xy[0][0]
            dy = seq_xy[1][1] - seq_xy[0][1]
        elif k == n - 1:
            dx = seq_xy[-1][0] - seq_xy[-2][0]
            dy = seq_xy[-1][1] - seq_xy[-2][1]
        else:
            dx = seq_xy[k + 1][0] - seq_xy[k - 1][0]
            dy = seq_xy[k + 1][1] - seq_xy[k - 1][1]
        if dx == 0.0 and dy == 0.0:
            headings[k] = last
        else:
            last = math.atan2(dy, dx)
            headings[k] = last
    return headings


def process_sequence(seq, anno_path, seq_dir, args):
    rows = load_mot_rows(anno_path)
    if not rows:
        print(f"  [skip] {seq}: no rows")
        return None

    m_per_px, n_calib, calib_src = calibrate_m_per_px(
        rows, args.assumed_car_len, args.default_mpp)

    size = read_frame_size(seq_dir)
    if size is None:
        # Fall back to the annotation bbox extent midpoint for the origin.
        max_u = max(r["left"] + r["w"] for r in rows)
        max_v = max(r["top"] + r["h"] for r in rows)
        u0, v0 = max_u / 2.0, max_v / 2.0
    else:
        W, H = size
        u0, v0 = W / 2.0, H / 2.0

    # 5-Hz downsample: keep every `step`-th source frame (0-indexed), renumber.
    step = max(1, int(round(args.fps / args.target_hz)))
    # VisDrone frames are 1-indexed; convert to 0-indexed before the % test.
    kept_frames = sorted({r["frame"] for r in rows if (r["frame"] - 1) % step == 0})
    fid_renumber = {f: i for i, f in enumerate(kept_frames)}

    # Per-agent ordered (renumbered_fid, x_m, y_m, group) over kept frames.
    per_agent = defaultdict(list)
    n_drop_class = 0
    for r in rows:
        if r["frame"] not in fid_renumber:
            continue
        g = group_for(r["cat"])
        if g is None:
            n_drop_class += 1
            continue
        cu = r["left"] + r["w"] / 2.0
        cv = r["top"] + r["h"] / 2.0
        # Scale-only pixel -> metric (ScaleOnlyShim convention: +y up, so flip v).
        x_m = (cu - u0) * m_per_px
        y_m = -(cv - v0) * m_per_px
        per_agent[r["tid"]].append((fid_renumber[r["frame"]], x_m, y_m, g))

    # Emit rows, computing per-agent finite-diff heading over its track.
    out_rows = []
    n_veh = n_vru = 0
    for tid, items in per_agent.items():
        items.sort(key=lambda t: t[0])
        xy = [(x, y) for (_, x, y, _) in items]
        headings = headings_from_positions(xy)
        grp = items[0][3]
        if grp == "VEHICLE/TARGET":
            n_veh += 1
        else:
            n_vru += 1
        for (fid, x, y, g), hd in zip(items, headings):
            out_rows.append((fid, tid, x, y, hd, g))

    out_rows.sort(key=lambda r: (r[0], r[1]))  # by frame, then aid

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{seq}.txt")
    with open(out_path, "w") as f:
        for fid, tid, x, y, hd, g in out_rows:
            f.write(f"{fid} {tid} {x:.4f} {y:.4f} {hd:.4f} {g}\n")

    flag = "  *** DEFAULT m/px (unreliable scale) ***" if calib_src == "default" else ""
    print(f"  [ok] {seq}: m/px={m_per_px:.5f} (calib={calib_src} n={n_calib})  "
          f"5Hz_frames={len(kept_frames)}  vehicles={n_veh} vrus={n_vru}  "
          f"rows={len(out_rows)}  -> {out_path}{flag}")
    return dict(seq=seq, m_per_px=m_per_px, n_calib=n_calib, calib_src=calib_src,
                n_frames_5hz=len(kept_frames), n_veh=n_veh, n_vru=n_vru,
                n_rows=len(out_rows))


def main():
    args = parse_args()
    anno_dir = os.path.join(args.root, "annotations")
    seq_dir_root = os.path.join(args.root, "sequences")
    if not os.path.isdir(anno_dir):
        print(f"ERROR: {anno_dir} not found", file=sys.stderr)
        return 2

    seqs = sorted(os.path.splitext(f)[0] for f in os.listdir(anno_dir)
                  if f.endswith(".txt"))
    if args.limit:
        seqs = seqs[:args.limit]
    print(f"[process_visdrone] {len(seqs)} sequence(s) under {args.root}")
    print(f"[process_visdrone] fps={args.fps} target_hz={args.target_hz} "
          f"step={max(1, int(round(args.fps / args.target_hz)))} -> {args.out}")

    summaries = []
    for seq in seqs:
        anno_path = os.path.join(anno_dir, f"{seq}.txt")
        seq_dir = os.path.join(seq_dir_root, seq)
        s = process_sequence(seq, anno_path, seq_dir, args)
        if s:
            summaries.append(s)

    if summaries:
        tot_rows = sum(s["n_rows"] for s in summaries)
        tot_veh = sum(s["n_veh"] for s in summaries)
        defaulted = [s["seq"] for s in summaries if s["calib_src"] == "default"]
        print(f"\n[process_visdrone] DONE: {len(summaries)} seq, "
              f"{tot_veh} vehicle tracks, {tot_rows} rows total.")
        if defaulted:
            print(f"[process_visdrone] WARNING: {len(defaulted)} seq used DEFAULT "
                  f"m/px (no car/van tracks) — flag/exclude from metrics: "
                  f"{', '.join(defaulted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
