"""
Synthetic moving-UAV clip generator (Week 3 Day 2-4 validation fixture).

Produces a short MP4 that looks like the inD_01 aerial scene shot from a
MOVING drone, plus a sidecar JSON with ground-truth per-frame homographies
and agent bounding boxes. Two consumers:

  1. `tests/test_ego_motion_shim.py` — uses the GT homographies to score
     `MovingCameraShim` recovery accuracy.
  2. Week 3 Day 4-5 end-to-end replay — uses the rendered video (which
     looks like real moving-UAV footage of inD vehicles) as the input
     clip for the full server-side ContextVAE pipeline.

How the clip is built
---------------------
- K0 = the un-padded inD_01 orthomap (cropped to remove the 202-px
  preprocessing pad). This is the canonical "what the drone would see if
  it were hovering exactly above the recording site at time t=0".
- For each synth frame t in [0, N_FRAMES), apply a known homography
  `H_K0 -> live[t]` (slow pan + 5 deg rotation + 10 % zoom over the clip)
  to K0 via `cv2.warpPerspective`. That's the moving-drone illusion.
- Overlay the inD agents at frame `t // FRAME_REPEAT`: pull their
  world-metric (x, y) from `data/levelx/val/inD_01.txt`, transform
  world -> K0-pixel via the static `PixelMetricShim`, then K0-pixel ->
  live-pixel via the SAME `H_K0 -> live[t]`. Draw a rectangle + agent id.
  Only `TARGET`-tagged agents are drawn (mirrors the training-time
  `inclusive_groups=["TARGET"]` filter in configs/levelx_train.py).
- Save MP4 via imageio (ffmpeg backend) + a JSON sidecar with one record
  per frame containing the GT `H_live -> K0` (inverse of the warp we
  applied) and the rendered agent bboxes in live-pixel `[l, t, w, h]`
  form (so a downstream tester could compare them against
  `MovingCameraShim`-reprojected positions).

Run
---
    python uav_guidance/server_code_only/main_program/synth_clip_gen.py
    # writes tmp/synth_uav_clip.mp4 + tmp/synth_uav_clip_gt.json

No CLI args; tweak the constants below if you need a different motion
profile or length.
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

sys.path.insert(0, str(_HERE))
from pixel_metric_shim import EXT_PAD, PixelMetricShim   # noqa: E402


# -- inputs --
MAP_PICKLE = _REPO_ROOT / "data" / "levelx" / "map" / "inD_01.pkl"
VAL_TXT = _REPO_ROOT / "data" / "levelx" / "val" / "inD_01.txt"

# -- outputs --
OUT_DIR = _REPO_ROOT / "tmp"
OUT_MP4 = OUT_DIR / "synth_uav_clip.mp4"
OUT_GT = OUT_DIR / "synth_uav_clip_gt.json"

# -- motion profile --
FPS = 25
N_FRAMES = 150                # ~6 s at 25 fps
FRAME_REPEAT = 5              # 1 inD 5-Hz frame per 5 synth 25-fps frames
PAN_TX_PX = 600               # total east translation by end of clip
PAN_TY_PX = 200               # total south translation
ROT_DEG = 5.0                 # total rotation
ZOOM_END = 1.10               # final scale

# -- agent overlay --
TARGET_GROUP_TOKEN = "TARGET"
BBOX_HALF_W = 12              # ~4 m at ~6.2 px/m
BBOX_HALF_H = 8               # rectangle, not oriented
BBOX_COLOR_BGR = (0, 255, 0)  # bright green
BBOX_THICKNESS = 2
ID_FONT_SCALE = 0.45
ID_FONT_THICKNESS = 1


def load_K0_bgr(map_pickle_path: Path) -> np.ndarray:
    """Decode the orthomap pickle into a BGR uint8 image at K0 (un-padded) scale.

    Pickle payload is `(semantic_map, H)` where semantic_map is (3, H_pad,
    W_pad) float32 in [-1, 1]. The static H in the pickle was fitted such
    that `H @ [x, y, 1] = [row_padded, col_padded, 1]` — see
    pixel_metric_shim.PixelMetricShim for the contract. K0 here is the
    image WITHOUT the EXT_PAD=202 border so that K0-pixel == un-padded
    image-pixel (the frame the four conversion methods speak natively)."""
    with open(map_pickle_path, "rb") as f:
        sm, _H = pickle.load(f)
    # (3, H, W) float32 in [-1, 1]  ->  (H, W, 3) uint8
    img_padded = ((sm.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    # The pickle is RGB (process_levelx.ipynb:425 uses Image.open(...).convert("RGB")
    # before normalizing to [-1, 1]). cv2.VideoWriter and the downstream
    # MovingCameraShim ORB matcher both expect BGR, so swap channels here.
    img_padded = cv2.cvtColor(img_padded, cv2.COLOR_RGB2BGR)
    img_K0 = img_padded[EXT_PAD:-EXT_PAD, EXT_PAD:-EXT_PAD]
    return np.ascontiguousarray(img_K0)


def read_inD_targets(val_txt_path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    """Read the preprocessed inD .txt and bucket TARGET-tagged rows by frame_id.

    Returns: {frame_id: [(agent_id, x_m, y_m), ...]}.
    Schema is `fid aid x y heading group_str` (see preprocessing notebook)."""
    by_frame: Dict[int, List[Tuple[int, float, float]]] = {}
    with open(val_txt_path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            aid = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            group = parts[5]
            if TARGET_GROUP_TOKEN not in group.split("/"):
                continue
            by_frame.setdefault(fid, []).append((aid, x, y))
    return by_frame


def make_h_trajectory(W: int, H: int, n_frames: int) -> List[np.ndarray]:
    """Per-frame H_K0 -> live. Smoothstep ramp from identity to the end pose."""
    cx, cy = W / 2.0, H / 2.0
    out: List[np.ndarray] = []
    for t in range(n_frames):
        u = t / max(n_frames - 1, 1)
        # smoothstep so frame 0 and frame N-1 have zero instantaneous velocity
        s = u * u * (3.0 - 2.0 * u)
        tx = PAN_TX_PX * s
        ty = PAN_TY_PX * s
        angle_deg = ROT_DEG * s
        scale = 1.0 + (ZOOM_END - 1.0) * s
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        H_full = np.eye(3, dtype=np.float64)
        H_full[:2, :] = M
        out.append(H_full)
    return out


def project_homog(H: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points; return (N, 2)."""
    pts_2d = np.asarray(pts_2d, dtype=np.float64).reshape(-1, 2)
    homog = np.concatenate([pts_2d, np.ones((pts_2d.shape[0], 1))], axis=1)
    out = (H @ homog.T).T
    return out[:, :2] / out[:, 2:3]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[synth] loading orthomap from {MAP_PICKLE}")
    K0 = load_K0_bgr(MAP_PICKLE)
    H_K0, W_K0 = K0.shape[:2]
    print(f"[synth] K0 size: {W_K0}x{H_K0} px")

    print(f"[synth] loading inD agents from {VAL_TXT}")
    targets_by_frame = read_inD_targets(VAL_TXT)
    n_inD_frames_needed = (N_FRAMES + FRAME_REPEAT - 1) // FRAME_REPEAT
    available = sorted(targets_by_frame.keys())
    print(
        f"[synth] {len(available)} inD frames with TARGETs available; "
        f"will use frames 0..{n_inD_frames_needed - 1}"
    )

    shim = PixelMetricShim(str(MAP_PICKLE))
    H_traj = make_h_trajectory(W_K0, H_K0, N_FRAMES)

    print(f"[synth] writing {N_FRAMES} frames at {FPS} fps -> {OUT_MP4}")
    # cv2.VideoWriter with mp4v fourcc; ffmpeg backend (no imageio_ffmpeg dep).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_MP4), fourcc, float(FPS), (W_K0, H_K0))
    if not writer.isOpened():
        print(f"[synth] FAIL: cv2.VideoWriter could not open {OUT_MP4}")
        return 1
    gt_frames: List[dict] = []

    try:
        for t in range(N_FRAMES):
            H_K0_to_live = H_traj[t]
            H_live_to_K0 = np.linalg.inv(H_K0_to_live)

            # Warp the orthomap into the live-camera viewpoint.
            live = cv2.warpPerspective(
                K0, H_K0_to_live, (W_K0, H_K0),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            inD_frame = t // FRAME_REPEAT
            agents_this_frame = targets_by_frame.get(inD_frame, [])
            rendered_agents: List[dict] = []

            for aid, x_m, y_m in agents_this_frame:
                # world -> K0 (PixelMetricShim already speaks un-padded
                # K0-pixel space) -> live via the warp we just applied.
                u_K0, v_K0 = shim.metric_to_pixel(x_m, y_m)
                uv_live = project_homog(
                    H_K0_to_live, np.array([[u_K0, v_K0]])
                )[0]
                u_live, v_live = float(uv_live[0]), float(uv_live[1])
                # Skip agents projected off-frame.
                if not (0 <= u_live < W_K0 and 0 <= v_live < H_K0):
                    continue
                x0 = int(round(u_live - BBOX_HALF_W))
                y0 = int(round(v_live - BBOX_HALF_H))
                x1 = int(round(u_live + BBOX_HALF_W))
                y1 = int(round(v_live + BBOX_HALF_H))
                cv2.rectangle(live, (x0, y0), (x1, y1), BBOX_COLOR_BGR, BBOX_THICKNESS)
                cv2.putText(
                    live, str(aid), (x0, max(0, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, ID_FONT_SCALE,
                    BBOX_COLOR_BGR, ID_FONT_THICKNESS, cv2.LINE_AA,
                )
                rendered_agents.append({
                    "aid": aid,
                    "ltwh_live": [x0, y0, x1 - x0, y1 - y0],
                    "center_live": [u_live, v_live],
                    "world": [x_m, y_m],
                })

            writer.write(live)
            gt_frames.append({
                "frame": t,
                "inD_frame": inD_frame,
                "H_live_to_K0": H_live_to_K0.tolist(),
                "agents": rendered_agents,
            })
            if (t + 1) % 25 == 0:
                print(f"[synth]   wrote frame {t + 1}/{N_FRAMES}")
    finally:
        writer.release()

    gt_blob = {
        "fps": FPS,
        "n_frames": N_FRAMES,
        "k0_shape_hw": [H_K0, W_K0],
        "frame_repeat": FRAME_REPEAT,
        "motion": {
            "pan_tx_px": PAN_TX_PX, "pan_ty_px": PAN_TY_PX,
            "rot_deg": ROT_DEG, "zoom_end": ZOOM_END,
        },
        "frames": gt_frames,
    }
    OUT_GT.write_text(json.dumps(gt_blob, indent=2))
    print(f"[synth] wrote GT sidecar -> {OUT_GT}")
    print(f"[synth] done. Video: {OUT_MP4.stat().st_size / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
