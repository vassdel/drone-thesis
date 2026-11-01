"""
Visual sanity check for the ContextVAE server-side inference path.

What it does
------------
1. Loads the un-padded inD_01 background PNG.
2. Reads inD_01.txt and picks one TARGET agent with a complete
   (OB_HORIZON + PRED_HORIZON) window (reusing verify_inference.py's helpers).
3. Runs `ContextVAEInferencer.infer` for 6 future positions in metric coords.
4. Projects them back to un-padded image pixels via `metrics_to_pixel`.
5. Overlays dots + polylines: green = 10 observations, yellow = 6 ground-truth
   future, red = 6 predicted.
6. Saves to `tmp/contextvae_visual_sanity.png`.

Why a separate script
---------------------
The smoke test checks only shape/magnitude on synthetic input, and
`verify_inference.py` checks numbers but renders nothing. This closes the loop:
a directional bug in the homography, ego-frame rotation, or K-mean collapse
makes the predicted dots land off-road or point backward — obvious at a glance,
but invisible to a numbers-only test if the magnitude happens to be plausible.

Run via
-------
    python uav_guidance/server_code_only/main_program/visual_sanity.py
    # writes tmp/contextvae_visual_sanity.png (relative to the repo root)
"""

import os
import sys

import numpy as np
# PIL rather than cv2: this offline overlay runs under the training conda env,
# which ships PIL but not OpenCV. The live server file still uses cv2 (it runs
# under the server env where OpenCV is available).
from PIL import Image, ImageDraw, ImageFont

# Sibling helper modules live alongside this script.
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
    # The un-padded background (before the EXT_PAD border) — what a stationary
    # replay assumes the UAV stream shows.
    bg_path = os.path.join(_REPO_ROOT, "levelx", "inD", "data", "01_background.png")
    ckpt = os.path.join(_REPO_ROOT, "tmp", "levelx_full_map_v1", "ckpt-best")
    cfg = os.path.join(_REPO_ROOT, "configs", "levelx_train.py")
    out_path = os.path.join(_REPO_ROOT, "tmp", "contextvae_visual_sanity.png")

    print(f"[viz] loading bg : {bg_path}")
    if not os.path.exists(bg_path):
        raise FileNotFoundError(f"Background image not found: {bg_path}")
    # Keep as a PIL Image to draw on via ImageDraw; .size is (W, H).
    img = Image.open(bg_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    print(f"[viz] bg size    : {img.size}")  # (W, H)

    print(f"[viz] reading val: {val_txt}")
    rows = _read_txt(val_txt)
    target_aid, frame_ids, target_xys_full, target_heading_world, frames_to_rows = (
        _find_target_window(rows)
    )
    print(f"[viz] target_aid : {target_aid}")
    print(f"[viz] frame range: {frame_ids[0]}..{frame_ids[-1]}")
    print(f"[viz] heading_w  : {target_heading_world:.4f} rad")
    target_history = target_xys_full[:OB_HORIZON]              # (10, 2)
    target_future_gt = target_xys_full[OB_HORIZON:]            # (25, 2)

    neighbors = _collect_neighbors(frame_ids, target_aid, frames_to_rows)
    print(f"[viz] neighbors  : {len(neighbors)}")

    # Pin map_model to resnet18 to match the headline checkpoint regardless of
    # the config file's current contents.
    inf = ContextVAEInferencer(
        ckpt_path=ckpt,
        config_path=cfg,
        map_pickle_path=map_pickle,
        model_overrides={"map_model": "resnet18"},
    )
    shim = PixelMetricShim(map_pickle)

    # Pass the stored first-obs-frame heading (radians, world frame), matching
    # the training loader. Omitting it falls back to an atan2 finite-diff,
    # rotating the crop by velocity direction rather than vehicle heading and
    # biasing the overlay.
    pred_metric = inf.infer(
        target_history, neighbors, heading_world=target_heading_world,
    )  # (6, 2) metric world
    print(f"[viz] pred metric:\n{pred_metric}")

    # Project everything to pixels for drawing: (N, 2) columns [u=col, v=row],
    # un-padded.
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
            # ImageDraw clips off-image coords, so no explicit bounds check.
            draw.ellipse(
                (ui - radius, vi - radius, ui + radius, vi + radius),
                fill=color,
            )

    _draw_dots(hist_uv, (0, 255, 0), radius=4)
    _draw_dots(gt_uv, (255, 255, 0), radius=4)
    _draw_dots(pred_uv, (255, 0, 0), radius=4)

    # Trace each sequence with a polyline so temporal order is visible.
    def _polyline(arr_uv, color, width=2):
        pts = [(int(round(u)), int(round(v))) for u, v in arr_uv]
        draw.line(pts, fill=color, width=width)

    _polyline(hist_uv, (0, 200, 0))
    _polyline(gt_uv, (200, 200, 0))
    _polyline(pred_uv, (200, 0, 0))

    # Legend box with the default PIL font (small but adequate here).
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
