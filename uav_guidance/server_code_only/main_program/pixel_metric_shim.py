"""
Pixel <-> metric world conversion shims.

ContextVAE operates in metric world coordinates (xCenter, yCenter in
levelXdata terms: local UTM-translated, +y up, meters). The server-side
pipeline operates in image pixel coordinates from a UAV video stream. These
shims bridge the two.

`PixelMetricShim` (stationary drone) converts via the homography H stored in
the map pickle from `preprocessing/process_levelx.ipynb`. The moving-camera
variant `ego_motion_shim.MovingCameraShim` exposes the same four-method API
plus an `update_frame(...)` lifecycle call and composes a static shim for its
K0 -> world step. `ScaleOnlyShim` (below) is the no-orthomap fallback for
datasets like VisDrone.

Coordinate frames
-----------------
1. World / metric: (x_m, y_m) meters, +y up, origin near the recording site.
   What ContextVAE consumes.
2. Un-padded image: (u, v) pixels of the original `XX_background.png`;
   u = column, v = row, +v down. Assumed to be what a stationary-drone replay
   frame shows.
3. Padded raster: (r_p, c_p) pixels of the pre-padded raster in the map
   pickle. Preprocessing pads the background by EXT_PAD = 202 px of -1.0 on
   all sides, and H is offset so it maps WORLD directly to padded-raster
   coords:
       padded_pixel = H @ [x_m, y_m, 1]
       world        = H_inv @ [r_p, c_p, 1]
       un_padded    = padded - EXT_PAD on both axes

Stationary-replay assumption
----------------------------
`PixelMetricShim` assumes the incoming frame IS the un-padded
`XX_background.png` — valid for an offline replay of an inD/uniD recording,
but not for live moving-camera footage. For moving cameras use
`MovingCameraShim`, which adds a per-frame homography into a stabilized frame.

Public API contract (identical across all shims)
-------------------------------------------------
    shim = PixelMetricShim(map_pickle_path)
    x_m, y_m = shim.pixel_to_metric(u, v)
    u, v     = shim.metric_to_pixel(x_m, y_m)
    xy_m  = shim.pixels_to_metric(uv_array)    # (N, 2) -> (N, 2)
    uv    = shim.metrics_to_pixel(xy_m_array)  # (N, 2) -> (N, 2)
"""

import pickle
from typing import Optional, Tuple

import numpy as np


# Padding (pixels) the preprocessing notebook adds around each background so
# a MAP_SIZE=224 crop still covers ~30 m post-rotation at the ~7 px/m scale.
# Must match the value used to build the pickle being loaded (same contract as
# EXT in contextvae_inference.py).
EXT_PAD = 202


class PixelMetricShim:
    """
    Two-way pixel <-> metric converter backed by a recording's
    preprocessed homography.

    Construct once per recording at server startup. The conversion is
    O(1) per point; batching just amortizes Python overhead.
    """

    def __init__(self, map_pickle_path: str, ext_pad: int = EXT_PAD):
        # The pickle is (semantic_map, H); only H is needed here. (Reading it
        # independently of the inferencer keeps the two modules decoupled.)
        with open(map_pickle_path, "rb") as f:
            _semantic_map, H = pickle.load(f)
        # float64 so the inverse doesn't accumulate error on round-trips.
        self.H = np.asarray(H, dtype=np.float64)
        if self.H.shape != (3, 3):
            raise ValueError(
                f"Map pickle H has unexpected shape {self.H.shape}; expected (3, 3)"
            )
        # H is a well-conditioned per-recording fit, so the inverse is safe.
        self.H_inv = np.linalg.inv(self.H)
        self.ext_pad = int(ext_pad)

    # ------------------------------------------------------------------
    # Single-point API
    # ------------------------------------------------------------------

    def pixel_to_metric(self, u: float, v: float) -> Tuple[float, float]:
        """
        Convert one un-padded image pixel `(u, v)` to world `(x_m, y_m)`.

        Shift into padded-raster coords (+EXT_PAD), then apply H_inv.

        Axis order: image (u, v) is (col, row), but H operates on (row, col, 1)
        — matching H @ [x, y, 1] = [row, col, 1] in contextvae/data.py — so we
        feed (v + EXT_PAD, u + EXT_PAD, 1) into H_inv.
        """
        r_p = v + self.ext_pad
        c_p = u + self.ext_pad
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

        Apply H -> padded-raster (row, col, 1); subtract EXT_PAD; reorder
        (row, col) -> (u=col, v=row).
        """
        r_p, c_p, w = self.H @ np.array([x_m, y_m, 1.0], dtype=np.float64)
        if abs(w) < 1e-12:
            raise RuntimeError(
                f"metric_to_pixel: degenerate homogeneous coord for "
                f"world ({x_m}, {y_m}) (w={w}). H may be malformed."
            )
        r_p /= w
        c_p /= w
        # Return floats (not ints) so callers can plot at sub-pixel positions.
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
        # Homogeneous (row, col, 1) in the padded frame: u = col, v = row.
        rc1 = np.stack(
            [uv[:, 1] + self.ext_pad, uv[:, 0] + self.ext_pad, np.ones(uv.shape[0])],
            axis=1,
        )  # (N, 3)
        out = np.einsum("ij,kj->ki", self.H_inv, rc1)  # (N, 3)
        # Divide by the homogeneous coord (no-op for affine H, safe if projective).
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

    For datasets with no orthomap / homography (e.g. VisDroneVDT-2018),
    `PixelMetricShim` cannot be used. A local-metric frame is still produced
    from:

      - a per-recording scalar `m_per_px` (calibrated from a known in-scene
        object length, typically a sedan at ~4.5 m);
      - an `origin_uv = (u0, v0)` pixel mapping to world (0, 0) — the first
        frame center is a clean "drone starts at origin" choice.

    The world frame is +y up (image-y is flipped in/out) to match the
    levelXdata convention the model trained on. Coordinates are unanchored to
    any global frame — meaningful only relative to other agents in the scene,
    which is all the encoder needs.

    Implements the same four-method API as `PixelMetricShim`, so the replay
    driver can swap them freely and `MovingCameraShim` can compose either as
    its static base via `base_shim=`.

    Limitations
    -----------
    A scalar m/px ignores perspective and altitude variation. For clips with
    significant zoom/altitude drift this is approximate — fine for qualitative
    figures, but do not report minADE numbers from a scale-only run.
    """

    def __init__(
        self,
        m_per_px: float,
        origin_uv: Tuple[float, float] = (0.0, 0.0),
    ):
        if m_per_px <= 0:
            raise ValueError(f"m_per_px must be positive; got {m_per_px}")
        self.m_per_px = float(m_per_px)
        # Cache the reciprocal so metric->pixel multiplies in the hot loop.
        self.px_per_m = 1.0 / self.m_per_px
        self.u0 = float(origin_uv[0])
        self.v0 = float(origin_uv[1])

    # ------------------------------------------------------------------
    # Single-point API
    # ------------------------------------------------------------------

    def pixel_to_metric(self, u: float, v: float) -> Tuple[float, float]:
        """Convert one un-padded image pixel `(u, v)` to world `(x_m, y_m)`."""
        # World x grows with u; world y grows with decreasing v (image-y down,
        # world-y up).
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
# Self-test: round-trip a known inD_01 world position through metric_to_pixel
# then pixel_to_metric; exercises the H / H_inv pair (and ScaleOnlyShim)
# without the inference model.
# Run via:  python uav_guidance/server_code_only/main_program/pixel_metric_shim.py
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

    # Known world position (first row of inD_01.txt).
    x_w, y_w = 132.0446, -31.9932

    u, v = shim.metric_to_pixel(x_w, y_w)
    print(f"[shim-test] ({x_w}, {y_w}) m -> ({u:.3f}, {v:.3f}) px (un-padded)")

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
    # Pure math, no fixture. Plausible VisDrone calibration: 1280x720 frame,
    # 0.05 m/px, origin at frame center.
    scale = ScaleOnlyShim(m_per_px=0.05, origin_uv=(640.0, 360.0))
    # World origin maps to the configured pixel origin.
    u0, v0 = scale.metric_to_pixel(0.0, 0.0)
    assert (u0, v0) == (640.0, 360.0), f"unexpected origin: {(u0, v0)}"
    # +1 m east -> +20 px (0.05 m/px = 20 px/m).
    u_e, v_e = scale.metric_to_pixel(1.0, 0.0)
    assert abs(u_e - 660.0) < 1e-9 and abs(v_e - 360.0) < 1e-9, (
        f"+1m east: expected (660, 360); got {(u_e, v_e)}"
    )
    # +1 m north -> -20 px in image v (image-y points down).
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
