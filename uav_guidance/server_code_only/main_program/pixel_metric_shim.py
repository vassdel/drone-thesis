"""
Pixel <-> metric world conversion shim.

Purpose
-------
ContextVAE operates in metric world coordinates (xCenter, yCenter in
levelXdata terminology — local UTM-translated, +y up, meters). The
Ntousis server-side pipeline operates in image pixel coordinates from
a UAV video stream.

This module bridges the two. For **Week 3 Day 1-2** (model swap; replay on
a STATIONARY drone clip derived from an inD/uniD recording), the bridge
is the homography `H` stored in the map pickle from
`preprocessing/process_levelx.ipynb`. The Day 2-4 moving-camera variant
lives in `ego_motion_shim.MovingCameraShim` — same four-method API plus
one new `update_frame(frame_bgr, mask_bboxes_ltwh)` lifecycle call. The
server file picks between the two via the `EGO_MOTION_ENABLED` flag near
the top of `scene_recog_socket_yolov8.py`. `PixelMetricShim` remains the
stationary-drone reference / fallback and is composed internally by
`MovingCameraShim` to handle the static K0 -> world step.

Coordinate frames involved
--------------------------
1. **World / metric frame.** `(x_m, y_m)` in meters. +y up. Origin near
   the recording site (per `recordingMeta.xUtmOrigin` / `yUtmOrigin`).
   This is what ContextVAE consumes.

2. **Un-padded image frame.** `(u, v)` in pixels of the original
   `XX_background.png` from the levelXdata recording. `u` is column
   index, `v` is row index. +v points DOWN (standard image convention).
   This is what we ASSUME the UAV video stream shows in Day 1-2 — i.e.
   a stationary drone hovering over the exact inD/uniD scene.

3. **Padded raster frame.** `(r_p, c_p)` in pixels of the pre-padded
   raster stored in `data/levelx/map/<recording>.pkl`. The preprocessing
   notebook pads the original background by `EXT_PAD = 202` pixels on
   all sides with constant `-1.0`. The homography `H` in the pickle was
   fitted in the un-padded frame and then OFFSET so it now maps WORLD
   to PADDED-RASTER pixel coords directly (per the build_map_pickle fix
   memory'd as `project_build_map_pickle_fix.md`).

   Therefore:
       padded_pixel = H @ [x_m, y_m, 1]
       world        = H_inv @ [r_p, c_p, 1]
       un_padded    = padded - EXT_PAD on both axes

The Day 1-2 stationary-replay assumption
----------------------------------------
For Day 1-2 we assume the UAV's incoming video frame IS the un-padded
`XX_background.png`. This is true for an offline replay of an inD/uniD
recording (or any test scenario where we render the background as the
UAV feed). For moving-camera live UAV footage this assumption breaks —
Day 2-4 ego-motion compensation will replace this shim with a per-frame
homography that maps the live UAV frame back into a stabilized world
frame.

Public API contract (preserved across Day 1-2 / 2-4 swap)
---------------------------------------------------------
    shim = PixelMetricShim(map_pickle_path)
    x_m, y_m = shim.pixel_to_metric(u, v)
    u, v     = shim.metric_to_pixel(x_m, y_m)
    # Batch versions take/return Nx2 ndarrays:
    xy_m  = shim.pixels_to_metric(uv_array)    # (N, 2) -> (N, 2)
    uv    = shim.metrics_to_pixel(xy_m_array)  # (N, 2) -> (N, 2)

The signatures are the same shape/dtype before and after Day 2-4. Only
the internals change.
"""

import pickle
from typing import Optional, Tuple

import numpy as np


# Mirror EXT_PAD from the preprocessing notebook (build_map_pickle).
# Memory `project_build_map_pickle_fix.md` (2026-05-05) records EXT_PAD=202
# as the value chosen so that MAP_SIZE=224 still covers ~30 m post-rotation
# at the canonical ~7 px/m scale. Hard-coded here for the same reason as
# `EXT` in `contextvae_inference.py`: it must match the value used to
# build the pickle being loaded, and re-deriving it would diverge if the
# notebook is ever re-run with a different choice.
EXT_PAD = 202


class PixelMetricShim:
    """
    Two-way pixel <-> metric converter backed by a recording's
    preprocessed homography.

    Construct once per recording at server startup. The conversion is
    O(1) per point; batching just amortizes Python overhead.
    """

    def __init__(self, map_pickle_path: str, ext_pad: int = EXT_PAD):
        # The map pickle is `(semantic_map_ndarray, H_ndarray)`. We only
        # need H here. Reading the pickle twice (once here, once in the
        # inferencer) is wasteful for a single recording but keeps the
        # two modules decoupled — they could be loaded from different
        # files in a future config where the orthomap and the homography
        # are stored separately.
        with open(map_pickle_path, "rb") as f:
            _semantic_map, H = pickle.load(f)
        # Cast to float64 for the matrix inverse: a single-precision H_inv
        # accumulates noticeable numerical error on long round-trips.
        self.H = np.asarray(H, dtype=np.float64)
        if self.H.shape != (3, 3):
            raise ValueError(
                f"Map pickle H has unexpected shape {self.H.shape}; expected (3, 3)"
            )
        # Pre-compute the inverse. The pickle's H is well-conditioned
        # (per-recording fit, no degenerate planes), so this is safe.
        self.H_inv = np.linalg.inv(self.H)
        self.ext_pad = int(ext_pad)

    # ------------------------------------------------------------------
    # Single-point API
    # ------------------------------------------------------------------

    def pixel_to_metric(self, u: float, v: float) -> Tuple[float, float]:
        """
        Convert one un-padded image pixel `(u, v)` to world `(x_m, y_m)`.

        Step 1: shift into padded-raster coords by adding EXT_PAD.
        Step 2: apply H_inv.

        Note the AXIS ORDER: image `(u, v)` is `(col, row)`. The homography
        operates on `(row, col, 1)` (matching the order
        `H @ [x, y, 1] = [row, col, 1]` that `contextvae/data.py:358`
        uses). So we feed (v + EXT_PAD, u + EXT_PAD, 1) into H_inv.
        """
        # u = column, v = row in un-padded frame; convert to padded.
        r_p = v + self.ext_pad
        c_p = u + self.ext_pad
        # Homogeneous solve. H_inv was fitted on (row, col, 1)
        # coordinates per data.py convention, so order matches here.
        x, y, w = self.H_inv @ np.array([r_p, c_p, 1.0], dtype=np.float64)
        if abs(w) < 1e-12:
            raise RuntimeError(
                f"pixel_to_metric: degenerate homogeneous coord for "
                f"pixel ({u}, {v}) (w={w}). H_inv may be malformed."
            )
        return float(x / w), float(y / w)

    def metric_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """
        Convert one world `(x_m, y_m)` to un-padded image pixel `(u, v)`.

        Step 1: apply H to get padded-raster `(row, col, 1)`.
        Step 2: subtract EXT_PAD to land in un-padded coords.
        Step 3: re-order `(row, col)` -> `(u=col, v=row)`.
        """
        r_p, c_p, w = self.H @ np.array([x_m, y_m, 1.0], dtype=np.float64)
        if abs(w) < 1e-12:
            raise RuntimeError(
                f"metric_to_pixel: degenerate homogeneous coord for "
                f"world ({x_m}, {y_m}) (w={w}). H may be malformed."
            )
        r_p /= w
        c_p /= w
        # Subtract pad offset to return to un-padded image coords.
        # We return floats (NOT ints) so callers can do sub-pixel work
        # (e.g. plot at fractional positions); int-cast is a UI concern.
        u = c_p - self.ext_pad
        v = r_p - self.ext_pad
        return float(u), float(v)

    # ------------------------------------------------------------------
    # Vectorized API for whole histories at once
    # ------------------------------------------------------------------

    def pixels_to_metric(self, uv: np.ndarray) -> np.ndarray:
        """
        Vectorized version of `pixel_to_metric`.

        Input  : ndarray of shape (N, 2) with columns [u, v] in un-padded pixels.
        Output : ndarray of shape (N, 2) with columns [x_m, y_m] in meters.
        """
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        # Build homogeneous (row, col, 1) coords in padded frame.
        # uv[:, 0] = u = col, uv[:, 1] = v = row.
        rc1 = np.stack(
            [uv[:, 1] + self.ext_pad, uv[:, 0] + self.ext_pad, np.ones(uv.shape[0])],
            axis=1,
        )  # (N, 3)
        # (3, 3) @ (3, N) -> (3, N). We use einsum for clarity.
        out = np.einsum("ij,kj->ki", self.H_inv, rc1)  # (N, 3)
        # Normalize by homogeneous coord. Affine H makes out[:, 2] == 1
        # identically, but we divide for safety in case H is projective.
        w = out[:, 2:3]
        if np.any(np.abs(w) < 1e-12):
            raise RuntimeError("pixels_to_metric: degenerate homogeneous coord in batch.")
        return out[:, :2] / w  # (N, 2) [x_m, y_m]

    def metrics_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """
        Vectorized version of `metric_to_pixel`.

        Input  : ndarray of shape (N, 2) with columns [x_m, y_m] in meters.
        Output : ndarray of shape (N, 2) with columns [u, v] in un-padded pixels.
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        # Build homogeneous (x, y, 1) coords.
        xyz = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1)  # (N, 3)
        out = np.einsum("ij,kj->ki", self.H, xyz)  # (N, 3) in (row, col, w)
        w = out[:, 2:3]
        if np.any(np.abs(w) < 1e-12):
            raise RuntimeError("metrics_to_pixel: degenerate homogeneous coord in batch.")
        rc = out[:, :2] / w  # (N, 2) (row, col) in padded frame
        # Reorder (row, col) -> (u=col, v=row); subtract pad offsets.
        u = rc[:, 1] - self.ext_pad
        v = rc[:, 0] - self.ext_pad
        return np.stack([u, v], axis=1)  # (N, 2) [u, v]


class ScaleOnlyShim:
    """
    Pixel <-> metric converter with a scalar meters-per-pixel factor.

    For datasets that ship NO orthomap / homography (e.g. VisDroneVDT-2018),
    we cannot use `PixelMetricShim`. We can still produce a local-metric
    coordinate system for the model by picking:

      - a per-recording scalar `m_per_px` (calibrated from the known length
        of an in-scene object — typically a sedan at ~4.5 m);
      - an `origin_uv = (u0, v0)` pixel location that maps to world (0, 0)
        — pick the FIRST frame center for the cleanest "drone starts at
        origin" setup.

    The world frame produced by this shim is **+y up** (image-y is flipped
    on the way in/out) to match the levelXdata convention the model was
    trained on. Absolute coordinates are unanchored to any global frame —
    they're meaningful only RELATIVE to other agents in the same scene,
    which is all the ContextVAE encoder needs.

    Implements the same four-method API as `PixelMetricShim`, so the
    replay driver can swap them with no other code changes, and
    `MovingCameraShim` can compose either as its static base via the
    `base_shim=` kwarg.

    Limitations
    -----------
    A scalar m/px ignores perspective distortion and altitude variation
    within the scene. For VisDrone clips with significant zoom / altitude
    drift this is approximate — acceptable for QUALITATIVE Ch6 figures
    but never publish `minADE` numbers off a scale-only run.
    """

    def __init__(
        self,
        m_per_px: float,
        origin_uv: Tuple[float, float] = (0.0, 0.0),
    ):
        if m_per_px <= 0:
            raise ValueError(f"m_per_px must be positive; got {m_per_px}")
        self.m_per_px = float(m_per_px)
        # Pre-compute px-per-meter so the metric->pixel direction does
        # a multiply (not a divide) in the hot loop.
        self.px_per_m = 1.0 / self.m_per_px
        self.u0 = float(origin_uv[0])
        self.v0 = float(origin_uv[1])

    # ------------------------------------------------------------------
    # Single-point API
    # ------------------------------------------------------------------

    def pixel_to_metric(self, u: float, v: float) -> Tuple[float, float]:
        """Convert one un-padded image pixel `(u, v)` to world `(x_m, y_m)`."""
        # u is image col, v is image row. World x grows with u, world y
        # grows with DECREASING v (image-y points down, world-y points up).
        x = (float(u) - self.u0) * self.m_per_px
        y = -(float(v) - self.v0) * self.m_per_px
        return x, y

    def metric_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Convert one world `(x_m, y_m)` to un-padded image pixel `(u, v)`."""
        u = self.u0 + float(x_m) * self.px_per_m
        v = self.v0 - float(y_m) * self.px_per_m  # y-flip
        return u, v

    # ------------------------------------------------------------------
    # Vectorized API for whole histories at once
    # ------------------------------------------------------------------

    def pixels_to_metric(self, uv: np.ndarray) -> np.ndarray:
        """Vectorized `pixel_to_metric`. `uv` is (N, 2) of [u, v]; returns (N, 2)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        x = (uv[:, 0] - self.u0) * self.m_per_px
        y = -(uv[:, 1] - self.v0) * self.m_per_px
        return np.stack([x, y], axis=1)

    def metrics_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """Vectorized `metric_to_pixel`. `xy` is (N, 2) of [x, y]; returns (N, 2)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        u = self.u0 + xy[:, 0] * self.px_per_m
        v = self.v0 - xy[:, 1] * self.px_per_m
        return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Self-test: round-trip a known agent position from the val .txt file.
# Run via:  python uav_guidance/server_code_only/main_program/pixel_metric_shim.py
#
# We pick the FIRST agent in inD_01.txt, take its first-obs-frame world
# position (132.0446, -31.9932), and check that metric_to_pixel followed by
# pixel_to_metric returns the same world coord to within float precision.
# This is the strongest test of the H / H_inv pair without needing the
# inference model.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import os

    HERE = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    pickle_path = os.path.join(repo, "data", "levelx", "map", "inD_01.pkl")
    print(f"[shim-test] loading {pickle_path}")

    shim = PixelMetricShim(pickle_path)
    print(f"[shim-test] H =\n{shim.H}")
    print(f"[shim-test] EXT_PAD = {shim.ext_pad}")

    # Known agent position from the val .txt — first row of inD_01.txt.
    x_w, y_w = 132.0446, -31.9932

    # World -> pixel
    u, v = shim.metric_to_pixel(x_w, y_w)
    print(f"[shim-test] ({x_w}, {y_w}) m -> ({u:.3f}, {v:.3f}) px (un-padded)")

    # Round-trip back
    x2, y2 = shim.pixel_to_metric(u, v)
    print(f"[shim-test] -> back to ({x2:.6f}, {y2:.6f}) m")

    err = np.hypot(x2 - x_w, y2 - y_w)
    print(f"[shim-test] round-trip error: {err:.2e} m")
    assert err < 1e-6, f"round-trip error {err} exceeds 1e-6"

    # Vectorized version should agree with single-point version.
    xy = np.array([[x_w, y_w], [x_w + 1, y_w], [x_w, y_w + 1]])
    uv = shim.metrics_to_pixel(xy)
    xy_back = shim.pixels_to_metric(uv)
    err_batch = np.linalg.norm(xy_back - xy, axis=-1).max()
    print(f"[shim-test] batched round-trip max error: {err_batch:.2e} m")
    assert err_batch < 1e-6, f"batched round-trip error {err_batch} exceeds 1e-6"

    # Sanity-check: a +1 m step in world should yield a non-trivial pixel step.
    step_px = np.linalg.norm(uv[1] - uv[0])
    print(f"[shim-test] +1 m in world == {step_px:.2f} px in image (sanity)")
    assert 1.0 < step_px < 50.0, (
        f"unrealistic px-per-meter scale {step_px} — expected 5-15 px/m for inD"
    )

    # ----- ScaleOnlyShim round-trip ----------------------------------------
    # No external fixture needed — pure math. Pick a plausible VisDrone
    # calibration: 1280x720 frame, ~0.05 m/px (typical mid-altitude UAV),
    # origin at frame center.
    scale = ScaleOnlyShim(m_per_px=0.05, origin_uv=(640.0, 360.0))
    # World origin should land at the configured pixel origin.
    u0, v0 = scale.metric_to_pixel(0.0, 0.0)
    assert (u0, v0) == (640.0, 360.0), f"unexpected origin: {(u0, v0)}"
    # +1 m in world x should move +20 px (0.05 m/px -> 20 px/m).
    u_e, v_e = scale.metric_to_pixel(1.0, 0.0)
    assert abs(u_e - 660.0) < 1e-9 and abs(v_e - 360.0) < 1e-9, (
        f"+1m east: expected (660, 360); got {(u_e, v_e)}"
    )
    # +1 m in world y should move -20 px in image v (image-y points down).
    u_n, v_n = scale.metric_to_pixel(0.0, 1.0)
    assert abs(u_n - 640.0) < 1e-9 and abs(v_n - 340.0) < 1e-9, (
        f"+1m north: expected (640, 340); got {(u_n, v_n)}"
    )
    # Round-trip a batch of points.
    pts_m = np.array([[0, 0], [1, 0], [0, 1], [-3, 5]], dtype=np.float64)
    pts_px = scale.metrics_to_pixel(pts_m)
    pts_m_back = scale.pixels_to_metric(pts_px)
    err_batch = np.linalg.norm(pts_m_back - pts_m, axis=-1).max()
    print(f"[shim-test] ScaleOnlyShim round-trip max err: {err_batch:.2e} m")
    assert err_batch < 1e-9, f"ScaleOnly round-trip err {err_batch}"

    print("[shim-test] OK")


if __name__ == "__main__":
    _self_test()
