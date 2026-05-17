"""
Visual sanity check for the ContextVAE inference path used by the
Ntousis server-side pipeline.

What it does
------------
1. Loads the un-padded inD_01 background PNG (the same image the
   preprocessing notebook started from before padding by EXT_PAD).
2. Reads inD_01.txt to pick one TARGET agent with a complete
   (OB_HORIZON + PRED_HORIZON) window — same agent_id=0 used in
   `verify_inference.py`.
3. Runs `ContextVAEInferencer.infer` to produce 6 future positions
   in metric coords.
4. Uses `PixelMetricShim.metrics_to_pixel` to project those positions
   back to un-padded image pixels.
5. Draws three overlays on the background:
     - GREEN dots  : the 10 observation positions.
     - YELLOW dots : the next 6 ground-truth positions (truth).
     - RED dots    : the 6 predicted positions.
6. Saves to `tmp/contextvae_visual_sanity.png`.

Why a separate script (not part of the smoke / verify ones)
-----------------------------------------------------------
- `_smoke_test` (in contextvae_inference.py) only verifies shape and
  magnitude on synthetic input — no map registration, no visualization.
- `verify_inference.py` checks numerical agreement with ground truth on
  a real val sample but doesn't render anything visual.
- This script closes the loop: if the homography, ego-frame rotation, or
  K-mean collapse is wrong DIRECTIONALLY, the predicted dots will land
  off-road or pointing backward, which a one-glance image inspection
  catches immediately. Numerical-only tests can miss directional bugs
  that happen to land in the right magnitude.

Run via
-------
    python uav_guidance/server_code_only/main_program/visual_sanity.py

Output
------
    Writes `tmp/contextvae_visual_sanity.png` (relative to the repo root).
"""

import os
import sys

import numpy as np
# We use PIL instead of cv2 because the contextvae conda env
# (`new_xupei_env`) does NOT ship OpenCV — that dependency lives in the
# server's own conda env, not the training env. PIL is part of the
# training env and is enough for offline overlay rendering. The real
# server file (`scene_recog_socket_yolov8.py`) still uses cv2 because it
# runs under the SERVER env where OpenCV is available.
from PIL import Image, ImageDraw, ImageFont

# Pull in the helper modules. Each lives alongside this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from contextvae_inference import (  # noqa: E402
    ContextVAEInferencer,
    OB_HORIZON,
    PRED_HORIZON,
    PROTOCOL_FUTURE_STEPS,
    _REPO_ROOT,
)
from pixel_metric_shim import PixelMetricShim  # noqa: E402
from verify_inference import (  # noqa: E402
    _read_txt,
    _find_target_window,
    _collect_neighbors,
)


def main() -> None:
    val_txt = os.path.join(_REPO_ROOT, "data", "levelx", "val", "inD_01.txt")
    map_pickle = os.path.join(_REPO_ROOT, "data", "levelx", "map", "inD_01.pkl")
    # The un-padded background — the same image the preprocessing notebook
    # started from before adding the EXT_PAD=202 border of -1.0. This is
    # what we ASSUME the UAV's video stream would show in a Day 1-2
    # stationary replay.
    bg_path = os.path.join(_REPO_ROOT, "levelx", "inD", "data", "01_background.png")
    ckpt = os.path.join(_REPO_ROOT, "tmp", "levelx_full_map_v1", "ckpt-best")
    cfg = os.path.join(_REPO_ROOT, "configs", "levelx_train.py")
    out_path = os.path.join(_REPO_ROOT, "tmp", "contextvae_visual_sanity.png")

    print(f"[viz] loading bg : {bg_path}")
    if not os.path.exists(bg_path):
        raise FileNotFoundError(f"Background image not found: {bg_path}")
    # PIL gives us (W, H) from .size; we keep it as a PIL Image to draw
    # onto via ImageDraw, then save as PNG at the end.
    img = Image.open(bg_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    print(f"[viz] bg size    : {img.size}")  # (W, H)

    print(f"[viz] reading val: {val_txt}")
    rows = _read_txt(val_txt)
    target_aid, frame_ids, target_xys_full, frames_to_rows = _find_target_window(rows)
    print(f"[viz] target_aid : {target_aid}")
    print(f"[viz] frame range: {frame_ids[0]}..{frame_ids[-1]}")
    target_history = target_xys_full[:OB_HORIZON]              # (10, 2)
    target_future_gt = target_xys_full[OB_HORIZON:]            # (25, 2)

    neighbors = _collect_neighbors(frame_ids, target_aid, frames_to_rows)
    print(f"[viz] neighbors  : {len(neighbors)}")

    # Inferencer + shim. Pin map_model to resnet18 to match the
    # headline checkpoint (independent of the local config's current
    # ablation toggles).
    inf = ContextVAEInferencer(
        ckpt_path=ckpt,
        config_path=cfg,
        map_pickle_path=map_pickle,
        model_overrides={"map_model": "resnet18"},
    )
    shim = PixelMetricShim(map_pickle)

    # Inference
    pred_metric = inf.infer(target_history, neighbors)  # (6, 2) metric world
    print(f"[viz] pred metric:\n{pred_metric}")

    # Project everything to pixel coords for drawing.
    # Each is shape (N, 2) with columns [u (col), v (row)] in un-padded pixels.
    hist_uv = shim.metrics_to_pixel(target_history)
    gt_uv = shim.metrics_to_pixel(target_future_gt[:PROTOCOL_FUTURE_STEPS])
    pred_uv = shim.metrics_to_pixel(pred_metric)

    # Draw overlays. Colors are in RGB (PIL convention — NOT BGR like cv2):
    #   - Green  : history (observation)
    #   - Yellow : ground truth future
    #   - Red    : prediction
    def _draw_dots(arr_uv, color, radius=4):
        for u, v in arr_uv:
            ui, vi = int(round(u)), int(round(v))
            # PIL's ellipse takes a bounding box. ImageDraw silently
            # clips coords that fall outside the image, so we don't
            # need to bounds-check explicitly.
            draw.ellipse(
                (ui - radius, vi - radius, ui + radius, vi + radius),
                fill=color,
            )

    _draw_dots(hist_uv, (0, 255, 0), radius=4)
    _draw_dots(gt_uv, (255, 255, 0), radius=4)
    _draw_dots(pred_uv, (255, 0, 0), radius=4)

    # Trace each sequence with a polyline so the temporal order is visible
    # at a glance. PIL's line takes a flat list of [x0, y0, x1, y1, ...]
    # tuples — we pass an iterable of (x, y) which works equivalently.
    def _polyline(arr_uv, color, width=2):
        pts = [(int(round(u)), int(round(v))) for u, v in arr_uv]
        draw.line(pts, fill=color, width=width)

    _polyline(hist_uv, (0, 200, 0))
    _polyline(gt_uv, (200, 200, 0))
    _polyline(pred_uv, (200, 0, 0))

    # Legend box. Default PIL font (no font file needed) is small but
    # adequate for a sanity-check overlay.
    draw.rectangle((10, 10, 260, 90), fill=(255, 255, 255))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((20, 20), "history (10 obs)", fill=(0, 150, 0), font=font)
    draw.text((20, 40), "ground truth (6 future)", fill=(150, 150, 0), font=font)
    draw.text((20, 60), "prediction (6 future)", fill=(150, 0, 0), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print(f"[viz] wrote {out_path}")

    # Numerical summary echoed for convenience.
    diffs = pred_metric - target_future_gt[:PROTOCOL_FUTURE_STEPS]
    per_step_err = np.linalg.norm(diffs, axis=-1)
    print(f"[viz] per-step error (m): {per_step_err}")
    print(f"[viz] ADE (6 steps, mean-of-K vs GT, m): {per_step_err.mean():.4f}")
    print(f"[viz] FDE (6 steps, mean-of-K vs GT, m): {per_step_err[-1]:.4f}")


if __name__ == "__main__":
    main()
