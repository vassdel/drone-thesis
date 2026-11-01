"""
ContextVAE single-target streaming inference for the UAV server pipeline.

Wraps a trained ContextVAE checkpoint so the server-side pipeline
(`scene_recog_socket_yolov8.py`, adapted from Ntousis' uav_guidance server)
can call it on per-frame DeepSORT track histories and receive K-sample future
trajectories. This replaces the LSTM predictor (`tr_pred_lstm`) in the
original server.

Why a dedicated inferencer rather than the training Dataloader
--------------------------------------------------------------
The training-side `contextvae.data.Dataloader` is a batch-oriented torch
Dataset that reads on-disk `.txt`/`.info` files and per-pickle map rasters
through a process-pool loader — none of which fits server-side inference,
where we have a single target track (batch N=1), history from a streaming
DeepSORT buffer, and a known recording whose map pickle is loaded once at
startup. The loader's per-item map crop and heading-rotation logic is still
needed, so it is re-implemented here in a stripped-down single-agent form.

Coordinate-frame contract
-------------------------
This module operates ENTIRELY in metric world coordinates; pixel<->metric
conversion is the caller's job (see `pixel_metric_shim.py` /
`ego_motion_shim.py`). The stationary-drone shim uses a constant
per-recording `orthoPxToMeter` scale; the moving-camera shim swaps in an
ego-motion-compensated homography but keeps the same metric output, so this
module is unaffected by which shim the caller uses.

ContextVAE conventions assumed here
-----------------------------------
1. Observation horizon OB_HORIZON = 10 frames at 5 Hz (2 s).
2. Prediction horizon PRED_HORIZON = 25 frames at 5 Hz (5 s).
3. Observation radius OB_RADIUS = 30 m (radius-based neighbor filter).
4. Map raster pre-baked to (3, H, W) in [-1, 1] with a 3x3 homography H
   mapping world (x, y) -> image (row, col). Pickle format matches
   `contextvae/data.py`'s `load_map`: a `(semantic_map_ndarray, H_ndarray)`
   tuple.
5. The neighbor tensor must span the full L1 + L2 window (35 frames). Future
   neighbor positions are unknown at inference, so frames
   [OB_HORIZON .. OB_HORIZON+PRED_HORIZON) are padded with 1e9; the encoder's
   radius mask (dist <= ob_radius) filters them. Zero-padding is wrong — it
   silently passes the mask and corrupts social attention.
6. Localization: the whole scene (target + neighbors) is rotated around the
   target's first-obs-frame position (x0, y0) by -heading. target[0] stays
   at (x0, y0); target[1:] is rotated. World coordinates are preserved — the
   origin is NOT translated to (x0, y0).
7. K samples (default K = 5) are drawn from the timewise stochastic decoder.
   `infer()` collapses them to a single trajectory (mean over K) to fit the
   server's 6-waypoint wire protocol; `infer_samples()` returns all K for
   multi-modal rendering.

Public API
----------
    inferencer = ContextVAEInferencer(
        ckpt_path="tmp/levelx_full_map_v1/ckpt-best",
        config_path="configs/levelx_train.py",
        map_pickle_path="data/levelx/map/inD_01.pkl",
        device="cuda",
    )
    future_xy = inferencer.infer(
        target_history_metric,      # (10, 2) (x, y) in metric world coords
        neighbor_histories_metric,  # {track_id: (10, 2) ndarray}
        heading_world=0.0,          # radians, world frame; matches the loader
    )  # -> (PROTOCOL_FUTURE_STEPS, 2) mean over K, in metric world coords.

Pass `heading_world` whenever a real heading is available; omitting it falls
back to a finite-difference estimate that degrades for stationary agents (see
`infer()`). The caller converts the result back to pixels with the same shim
it used on the inputs.
"""

import os
import sys
import math
import pickle
import warnings
import importlib.util
from typing import Dict, Optional

import numpy as np
import torch

# Make the project's `contextvae` package importable from this server-side
# subdirectory without a pip/editable install (the package has no setup.py).
# Climb three levels to the repo root and prepend it to sys.path.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from contextvae.model import ContextVAE  # noqa: E402


# ---------------------------------------------------------------------------
# Constants. These MUST match the training config (configs/levelx_train.py)
# the checkpoint was produced with. They are hard-coded (not config-driven)
# so a config edited for a later ablation cannot silently mis-shape the
# model's inputs relative to the loaded weights.
# ---------------------------------------------------------------------------

# Past frames the encoder consumes: 10 at 5 Hz = 2 s.
OB_HORIZON = 10

# Future frames the decoder produces: 25 at 5 Hz = 5 s.
PRED_HORIZON = 25

# Radius-based neighbor filter (meters). The encoder's distance mask drops
# neighbors beyond this range at every frame.
OB_RADIUS = 30.0

# Square edge of the final map crop fed to MapEncode (224 x 224 RGB).
MAP_SIZE = 224

# Half-extent of the pre-rotation map crop (pixels), matching `self.EXT` in
# contextvae/data.py (ceil(MAP_SIZE/4 * sqrt(13)) = 202 for MAP_SIZE=224).
# Hard-coded to the training-time value so the crop geometry matches the
# checkpoint. The pre-rotation patch is (3, 2*EXT, 2*EXT) = (3, 404, 404);
# after rotation it is sliced to (3, 224, 224) via the offsets below.
EXT = 202

# Asymmetric crop offsets applied after rotation. LEFT < RIGHT is intentional:
# once rotated, the agent's forward heading points along +x, and we want more
# map context ahead of the agent than behind (56 px behind, 168 px ahead).
# Mirrors the TOP/BOTTOM/LEFT/RIGHT slice in contextvae/data.py, offset by EXT
# to index into the 2*EXT-wide rotated patch.
_CROP_TOP    = EXT - MAP_SIZE // 2          # 90
_CROP_BOTTOM = EXT + (MAP_SIZE - MAP_SIZE // 2)  # 314
_CROP_LEFT   = EXT - MAP_SIZE // 4          # 146
_CROP_RIGHT  = EXT + (MAP_SIZE - MAP_SIZE // 4)  # 370

# Default number of decoder samples, matching the training run's pred_samples
# so server-side latency tracks the offline eval.
DEFAULT_K_SAMPLES = 5

# Future steps returned to the UAV over the socket protocol. The original LSTM
# returned 6; ContextVAE's 25-step output is truncated to the first 6 to keep
# the wire protocol byte-compatible with the onboard guidance interface.
PROTOCOL_FUTURE_STEPS = 6

# Sentinel for absent neighbor slots. Chosen so the radius mask
# (dist <= ob_radius) filters them out; zero-padding would place fake
# neighbors at the origin and corrupt social attention.
NEIGHBOR_PAD_VALUE = 1.0e9

# Hard cap on neighbor slots. Attention cost scales with Nn, so this bounds
# per-frame cost in crowded scenes. Loose enough not to truncate typical
# inD/uniD intersections.
MAX_NEIGHBORS = 12


class ContextVAEInferencer:
    """
    Wraps a trained ContextVAE checkpoint for single-target streaming
    inference.

    One instance per server connection. The model and map raster are loaded
    once at construction; `infer(...)` / `infer_samples(...)` are then called
    per frame.

    Thread-safety: the underlying torch model is not thread-safe
    (flatten_parameters mutates state). Use one inferencer per worker thread,
    or guard the infer calls with an external mutex.
    """

    def __init__(
        self,
        ckpt_path: str,
        config_path: str,
        map_pickle_path: Optional[str] = None,
        device: Optional[str] = None,
        k_samples: int = DEFAULT_K_SAMPLES,
        model_overrides: Optional[Dict] = None,
    ):
        # `map_pickle_path=None` selects no-map (S-ATTN-only) mode: the map
        # pickle is skipped and `map=None` is passed to the model, which takes
        # ContextVAE.enc()'s no-map branch. The caller must match the
        # checkpoint by also passing `model_overrides={"map_model": None}`;
        # a mismatch is caught (loudly) by load_state_dict below.
        #
        # Default to CUDA when available (inference fits in <1 GB VRAM);
        # pass device="cpu" to test on machines without CUDA.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.k_samples = int(k_samples)

        # One-shot latch for the heading-fallback warning in `infer()`
        # (fired the first time a caller omits `heading_world`).
        self._heading_fallback_warned = False

        # --- Load the config file -------------------------------------------
        # Training-side configs are plain Python modules. Load via importlib
        # so we can read `config.model` (the ContextVAE constructor kwargs)
        # without hard-coding them, mirroring contextvae/main.py.
        spec = importlib.util.spec_from_file_location(
            "contextvae_inference_config",  # synthetic module name
            config_path,
            submodule_search_locations=[os.path.dirname(config_path)],
        )
        self._config = importlib.util.module_from_spec(spec)
        # Register in sys.modules before exec so configs that inherit via a
        # relative import (e.g. `from .levelx_train import *`) can resolve
        # their parent package.
        sys.modules[spec.name] = self._config
        spec.loader.exec_module(self._config)

        # --- Instantiate the model ------------------------------------------
        # config.model is the ContextVAE kwargs dict (horizon, ob_radius,
        # hidden_dim, map_model, ...). Since the config file is edited across
        # ablations, `model_overrides` lets callers pin the architecture to
        # match the checkpoint regardless of the file's current contents. A
        # mismatch surfaces as a shape error in load_state_dict below.
        model_kwargs = dict(self._config.model)
        if model_overrides:
            model_kwargs.update(model_overrides)
        self.model = ContextVAE(**model_kwargs)
        self._model_kwargs = model_kwargs  # kept for logging

        # --- Load the trained weights ---------------------------------------
        # The ckpt is a torch-pickled dict {"model", "ade", "fde", "epoch",
        # ...}; only "model" is needed for inference. map_location handles a
        # ckpt saved on a different device than we load on.
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        # eval() disables dropout (incl. the map-input dropout) and switches
        # BN to running stats; without it, training noise would leak into
        # every server-side prediction.
        self.model.eval()
        # Record ckpt metadata so startup logs state the loaded version.
        self._loaded_epoch = state.get("epoch", -1)
        self._loaded_ade = state.get("ade", float("nan"))
        self._loaded_fde = state.get("fde", float("nan"))

        # --- Load the map raster (optional) ---------------------------------
        # Pickle is `(semantic_map, H)` per contextvae/data.py's load_map:
        #   semantic_map: (3, H_full, W_full) in [-1, 1]
        #   H: (3, 3) mapping world (x, y, 1) -> image (row, col, 1)
        # The map is held on-device so the affine crop avoids a per-frame H2D
        # copy. When map_pickle_path is None, no-map mode is active (see note
        # in the __init__ signature above).
        if map_pickle_path is not None:
            with open(map_pickle_path, "rb") as f:
                sem_map_np, H_np = pickle.load(f)
            # Force float32 in case a future pickle ships another dtype.
            self._semantic_map = torch.tensor(
                sem_map_np, dtype=torch.float32, device=self.device
            )
            # Keep H on CPU as numpy: it is used only for a single
            # `H @ [x, y, 1]` per call — not worth a GPU round-trip.
            self._H = np.asarray(H_np, dtype=np.float64)

            # Fail loud at construction, not on the first infer() call.
            assert self._semantic_map.dim() == 3 and self._semantic_map.size(0) == 3, (
                f"map pickle has unexpected shape {self._semantic_map.shape}; "
                "expected (3, H, W)"
            )
            assert self._H.shape == (3, 3), (
                f"map pickle H has unexpected shape {self._H.shape}; expected (3, 3)"
            )
        else:
            # No-map mode: attributes still exist (None sentinel) so downstream
            # access doesn't raise; _run_forward checks for None before cropping.
            self._semantic_map = None
            self._H = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_forward(
        self,
        target_history_metric: np.ndarray,
        neighbor_histories_metric: Dict[int, np.ndarray],
        heading_world: Optional[float],
    ):
        """
        Shared forward pass for `infer()` and `infer_samples()`.

        Validates inputs, computes heading, builds the localization rotation
        and the model tensors, and calls the model. Returns the raw K-sample
        prediction in the LOCALIZED frame plus the (pivot, R) pair needed to
        un-localize. Callers handle K-collapse (or not), un-rotation, and any
        truncation. Splitting this out keeps the socket entry point `infer()`
        and the replay entry point `infer_samples()` on identical model-input
        construction.

        Returns
        -------
        pred : np.ndarray, shape (K, PRED_HORIZON, 2), dtype float64
            K future trajectories in the LOCALIZED world frame (i.e. the
            same frame as the model's input tensors). The agent's `N=1`
            batch axis has been squeezed out.
        pivot_xy : np.ndarray, shape (2,), dtype float64
            The pivot used for localization (first-obs-frame target position).
        R : np.ndarray, shape (2, 2), dtype float64
            The rotation matrix applied during localization. Its transpose
            un-rotates back to the original world frame.
        """
        # --- 0. Validate inputs --------------------------------------------
        # Cheap shape check at the boundary; without it a bad shape would fail
        # later inside the encoder with a less informative error.
        target_history_metric = np.asarray(target_history_metric, dtype=np.float64)
        if target_history_metric.shape != (OB_HORIZON, 2):
            raise ValueError(
                f"target_history_metric must have shape ({OB_HORIZON}, 2); "
                f"got {target_history_metric.shape}"
            )

        # --- 1. Compute heading at the first obs frame ---------------------
        # Training-time localization uses the heading at the FIRST obs frame.
        # Preferred: the caller supplies `heading_world` (radians, world
        # frame) from the same source training used — the preprocessed .txt
        # heading column, or a DeepSORT bbox-displacement estimate over a
        # multi-step window. Used verbatim to match training.
        #
        # Fallback (heading_world is None): approximate via finite difference
        # atan2(v0_y, v0_x) over the first two obs positions. For
        # stationary/slow agents this collapses to 0, so the map crop is
        # axis-aligned to the world frame rather than the agent frame — a
        # distribution the M-ATTN encoder never saw, degrading predictions.
        # Warn once per instance; stay quiet afterward to avoid log spam.
        if heading_world is not None:
            heading = float(heading_world)
        else:
            if not self._heading_fallback_warned:
                warnings.warn(
                    "ContextVAEInferencer.infer() called without "
                    "`heading_world`; falling back to atan2(v0_y, v0_x) over "
                    "the first two obs positions. This DIVERGES from the "
                    "training-time loader (which reads the stored heading) "
                    "and degrades for stationary/slow agents (atan2 collapses "
                    "to 0, producing un-rotated map crops the M-ATTN encoder "
                    "never saw). Pass `heading_world` (radians, world frame) "
                    "to silence this warning. (Warning emitted once per "
                    "ContextVAEInferencer instance.)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._heading_fallback_warned = True
            v0 = target_history_metric[1] - target_history_metric[0]
            heading = math.atan2(v0[1], v0[0])  # radians, world frame

        # Training's loader stores the rotation angle as -heading; match the
        # sign so the rotation below mirrors it.
        angle = -heading

        # --- 2. Pivot for the localization rotation ------------------------
        # Pivot = first obs-frame position. After rotating the scene around
        # it, target[0] is unchanged and target[1:] is rotated; world
        # coordinates are preserved (origin is not translated to the pivot).
        pivot_xy = target_history_metric[0].copy()  # shape (2,)

        # --- 3. Apply the localization rotation ----------------------------
        # Subtract pivot -> rotate -> add pivot back: rotates about the pivot
        # while keeping world coordinates.
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)  # (2, 2)

        target_loc = self._rotate_around_pivot(target_history_metric, pivot_xy, R)

        # Rotate each neighbor history by the same R; kept as a dict here and
        # collated into the model tensor in step 5.
        neighbors_loc = {
            tid: self._rotate_around_pivot(
                np.asarray(hist, dtype=np.float64), pivot_xy, R
            )
            for tid, hist in neighbor_histories_metric.items()
        }

        # --- 4. Build the target's 6-D state tensor ------------------------
        # Model expects (OB_HORIZON, N=1, 6) with channels (x, y, vx, vy, ax,
        # ay). The encoder recomputes velocity from positions and reads only
        # the first acceleration (state[1, :, 4:6]); the other derived
        # channels are inert but filled for format parity and debuggability.
        target_state_6d = self._positions_to_6d_state(target_loc)  # (10, 6)
        x_tensor = torch.tensor(
            target_state_6d[:, None, :],  # add the N=1 batch dim
            dtype=torch.float32,
            device=self.device,
        )

        # --- 5. Build the neighbor tensor ----------------------------------
        # Shape (OB_HORIZON + PRED_HORIZON, N=1, Nn, 6). Future frames are
        # padded with NEIGHBOR_PAD_VALUE (unknown at inference); the encoder's
        # distance mask filters them. See _build_neighbor_tensor.
        neighbor_tensor = self._build_neighbor_tensor(
            target_loc=target_loc,
            neighbors_loc=neighbors_loc,
        )  # (35, 1, Nn, 6) float32

        # --- 6. Build the map tensor ---------------------------------------
        # Two-stage build mirroring contextvae/data.py: (a) crop a
        # (3, 2*EXT, 2*EXT) patch centered at the pivot pixel H @ (x0, y0, 1);
        # (b) rotate by `angle` via affine_grid/grid_sample, then slice to
        # (3, MAP_SIZE, MAP_SIZE) with the forward-biased asymmetric crop.
        # Output (N=1, 3, MAP_SIZE, MAP_SIZE), unsqueezed to the training-time
        # (1, N, 3, H, W) layout. In no-map mode we pass map_tensor=None and
        # the encoder takes its no-map branch (S-ATTN path unaffected).
        if self._semantic_map is not None:
            map_tensor = self._build_map_tensor(
                pivot_xy=pivot_xy, angle=angle
            )  # (1, 1, 3, 224, 224) float32
        else:
            map_tensor = None

        # --- 7. Call the model ---------------------------------------------
        # Eval-mode forward (x, neighbor, map, n_predictions=K) returns
        # (K, PRED_HORIZON, N, 2). seq_len is left default: at inference all
        # OB_HORIZON frames are present, so there is no pad-mode.
        pred = self.model(
            x_tensor,
            neighbor_tensor,
            map_tensor,
            n_predictions=self.k_samples,
        )
        # pred: (K, 25, 1, 2). The model already accumulates (cumsum) and
        # shifts by x_T, so pred is absolute positions in the SAME localized
        # frame as x_tensor (world coords rotated about pivot_xy by `angle`).

        # Drop the N=1 axis and convert to numpy float64 so all downstream
        # geometry stays in numpy, matching the rest of this module.
        pred_np = pred.squeeze(2).cpu().numpy().astype(np.float64)  # (K, 25, 2)
        return pred_np, pivot_xy, R

    def infer(
        self,
        target_history_metric: np.ndarray,
        neighbor_histories_metric: Dict[int, np.ndarray],
        heading_world: Optional[float] = None,
    ) -> np.ndarray:
        """
        Run one inference step and return the next `PROTOCOL_FUTURE_STEPS`
        future positions of the target (mean over K samples).

        This is the socket-protocol entry point. The wire protocol expects a
        single (6, 2) trajectory, so the K samples are mean-collapsed and
        truncated to the first 6 steps. For multi-modal rendering, use
        `infer_samples()` instead.

        Parameters
        ----------
        target_history_metric : np.ndarray, shape (OB_HORIZON, 2), dtype float
            The target's last 10 (x, y) positions in METRIC WORLD coordinates,
            ordered oldest -> newest. The last row is the position at "now".
        neighbor_histories_metric : Dict[int, np.ndarray]
            Mapping from DeepSORT track_id -> (OB_HORIZON, 2) array of the
            same form. The target track_id should NOT be in this dict.
            Empty dict is allowed (means no neighbors visible).
        heading_world : Optional[float], default None
            Target heading at the FIRST obs frame, in RADIANS, world frame
            (atan2-convention angle from +x). When provided, used verbatim for
            the localization rotation and map crop, matching the training-time
            loader (which reads the stored `heading`). When None, falls back to
            atan2(v0_y, v0_x) over the first two obs positions and emits a
            one-shot warning; for stationary/slow agents this collapses to 0
            and the M-ATTN encoder receives an un-rotated crop it never saw in
            training. Callers with a real heading (preprocessed .txt column, or
            DeepSORT bbox displacement over a longer history) should pass it.

        Returns
        -------
        future_xy : np.ndarray, shape (PROTOCOL_FUTURE_STEPS, 2), dtype float64
            The mean-over-K-samples future trajectory of the target, in
            METRIC WORLD coordinates, ordered nearest -> farthest in time.
            Step 0 corresponds to T + 0.2 s (one 5-Hz step ahead of "now").
        """
        pred_loc, pivot_xy, R = self._run_forward(
            target_history_metric=target_history_metric,
            neighbor_histories_metric=neighbor_histories_metric,
            heading_world=heading_world,
        )  # (K, 25, 2), (2,), (2, 2)

        # Mean-collapse over K and un-localize (R.T inverts the localization
        # rotation).
        pred_mean = pred_loc.mean(axis=0)  # (25, 2)
        R_inv = R.T
        pred_unloc = self._rotate_around_pivot(pred_mean, pivot_xy, R_inv)

        # Truncate 25 -> 6 for the wire protocol.
        return pred_unloc[:PROTOCOL_FUTURE_STEPS]

    def infer_samples(
        self,
        target_history_metric: np.ndarray,
        neighbor_histories_metric: Dict[int, np.ndarray],
        heading_world: Optional[float] = None,
    ) -> np.ndarray:
        """
        Run one inference step and return ALL K sample trajectories.

        This is the replay-driver entry point — used by offline tooling (e.g.
        `replay_video.py`) that needs the full multi-modal output rather than
        the mean-collapsed trajectory the socket protocol carries.

        Same input contract as `infer()`, but returns all K samples (no mean
        collapse) over the full PRED_HORIZON (25 steps = 5 s) rather than the
        protocol-truncated 6 — there is no wire-protocol constraint here.

        Returns
        -------
        future_xy_samples : np.ndarray, shape (K, PRED_HORIZON, 2), float64
            K predicted future trajectories of the target in METRIC WORLD
            coordinates, ordered nearest -> farthest in time. `K` equals
            `self.k_samples` (DEFAULT_K_SAMPLES = 5 unless overridden at
            construction).
        """
        pred_loc, pivot_xy, R = self._run_forward(
            target_history_metric=target_history_metric,
            neighbor_histories_metric=neighbor_histories_metric,
            heading_world=heading_world,
        )  # (K, 25, 2), (2,), (2, 2)

        # Un-localize all K trajectories at once (_rotate_around_pivot handles
        # arbitrary leading dims via einsum).
        R_inv = R.T
        pred_unloc = self._rotate_around_pivot(pred_loc, pivot_xy, R_inv)
        return pred_unloc  # (K, 25, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_around_pivot(
        points_xy: np.ndarray, pivot_xy: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        """
        Rotate a (..., 2) array of points about `pivot_xy` by the 2x2 matrix
        `R`, preserving world coordinates (subtract pivot, rotate, add back).

        Shapes
        ------
        points_xy : (..., 2) — any leading dimensions allowed
        pivot_xy  : (2,)
        R         : (2, 2)
        returns   : same shape as points_xy
        """
        centered = points_xy - pivot_xy
        # einsum applies R @ p along the last axis without reshaping.
        rotated = np.einsum("ij,...j->...i", R, centered)
        return rotated + pivot_xy

    @staticmethod
    def _positions_to_6d_state(positions: np.ndarray) -> np.ndarray:
        """
        Convert an (OB_HORIZON, 2) array of (x, y) positions into the
        (OB_HORIZON, 6) state tensor `(x, y, vx, vy, ax, ay)` that
        ContextVAE expects.

        Velocity and acceleration are first-order finite differences in units
        of meters-per-timestep (NOT m/s), matching the training convention in
        contextvae/data.py's `extend()`. (Converting to m/s requires *5 at the
        5 Hz cadence — relevant only for the LSTM-baseline comparison.)

        Cold-start handling mirrors the loader: first-frame velocity copies
        frame 1 (v[0] == v[1]), and the first two accelerations copy frame 2
        (a[0] == a[1] == a[2]), since lower-order finite differences at the
        start of a trajectory are undefined.

        The encoder recomputes velocity from positions and reads only the
        first acceleration (state[1, 4:6]); the other derived channels are
        inert for the target but filled for format parity and debuggability.
        """
        T = positions.shape[0]
        out = np.zeros((T, 6), dtype=np.float64)
        out[:, 0:2] = positions

        # Velocity v_t = pos_t - pos_{t-1}, with v[0] := v[1].
        v = np.zeros_like(positions)
        v[1:] = np.diff(positions, axis=0)
        v[0] = v[1]  # cold-start copy
        out[:, 2:4] = v

        # Acceleration a_t = v_t - v_{t-1}, with a[0] := a[1] := a[2].
        a = np.zeros_like(positions)
        if T >= 3:
            a[2:] = np.diff(v[1:], axis=0)  # fills rows 2..T-1
            a[1] = a[2]
            a[0] = a[2]
        elif T >= 2:
            # Defensive: should not happen at the validated OB_HORIZON=10.
            a[:] = 0.0
        out[:, 4:6] = a

        return out

    def _build_neighbor_tensor(
        self,
        target_loc: np.ndarray,
        neighbors_loc: Dict[int, np.ndarray],
    ) -> torch.Tensor:
        """
        Build the neighbor tensor of shape (OB_HORIZON + PRED_HORIZON, 1, Nn, 6).

        Contract
        --------
        - Frames [0 .. OB_HORIZON) carry actual neighbor histories (6-D state,
          same convention as the target).
        - Frames [OB_HORIZON .. OB_HORIZON+PRED_HORIZON) are filled with
          NEIGHBOR_PAD_VALUE; the encoder's distance mask filters them.
        - Neighbors beyond OB_RADIUS of the target at every obs frame are
          pre-filtered here so the encoder doesn't spend budget on them.
        - The slot dimension Nn is padded with all-1e9 entries up to
          MAX_NEIGHBORS, matching the loader's collate-time padding.

        Why the full L1+L2 window even at inference
        -------------------------------------------
        At training the posterior RNN consumes neighbor states over the future
        horizon. At inference only the prior branch runs (which never touches
        the future slice), but the forward path still slices neighbor[1:], so
        the tensor must physically have 35 frames or enc() crashes on shape.
        """
        total_frames = OB_HORIZON + PRED_HORIZON  # 35

        # Step 1: radius-filter neighbors using the localized positions. Keep a
        # neighbor if it is within OB_RADIUS of the target at ANY obs frame.
        kept_states: list[np.ndarray] = []
        for _tid, hist in neighbors_loc.items():
            # Per-obs-frame target<->neighbor distance; both (OB_HORIZON, 2).
            dist = np.linalg.norm(hist - target_loc, axis=-1)
            if np.any(dist <= OB_RADIUS):
                kept_states.append(self._positions_to_6d_state(hist))
                if len(kept_states) >= MAX_NEIGHBORS:
                    break  # hard cap; rare in inD/uniD scenes

        # Step 2: assemble the full-window tensor. Initialize everything to the
        # pad value, then overlay the obs-window states; future frames stay
        # padded. Always emit MAX_NEIGHBORS slots (unused ones all-1e9) so the
        # last-dim shape is constant across calls.
        nb = np.full(
            (total_frames, 1, MAX_NEIGHBORS, 6),
            fill_value=NEIGHBOR_PAD_VALUE,
            dtype=np.float32,
        )
        for slot_idx, state6d in enumerate(kept_states):
            # Place each (OB_HORIZON, 6) state in its slot over the obs frames.
            nb[:OB_HORIZON, 0, slot_idx, :] = state6d.astype(np.float32)

        return torch.from_numpy(nb).to(self.device)

    def _build_map_tensor(
        self, pivot_xy: np.ndarray, angle: float
    ) -> torch.Tensor:
        """
        Build the map tensor of shape (1, N=1, 3, MAP_SIZE, MAP_SIZE).

        Two-stage construction mirrors contextvae/data.py:
          1. Pre-rotation pixel crop (3, 2*EXT, 2*EXT) centered at the agent's
             first-obs-frame pixel (r, c) = H @ (x, y, 1).
          2. Affine rotation by `angle` via affine_grid + grid_sample, then an
             asymmetric slice to (3, MAP_SIZE, MAP_SIZE).

        The leading 1 matches the (B, N, 3, H, W) layout the encoder expects;
        MapEncode.forward collapses B and N with .view().
        """
        # --- Stage 1: pre-rotation pixel crop ------------------------------
        # Homography maps the pivot world coords to the agent's (row, col) in
        # the full raster. For an affine H the w-divide is a no-op; kept for
        # safety against a projective pickle.
        x0, y0 = float(pivot_xy[0]), float(pivot_xy[1])
        r_f, c_f, w_f = self._H @ np.array([x0, y0, 1.0], dtype=np.float64)
        # Guard degenerate w=0 (impossible for a well-formed H; fail loud).
        if abs(w_f) < 1e-12:
            raise RuntimeError(
                f"Map homography produced w=0 for pivot ({x0}, {y0}). "
                "Pickle H may be malformed."
            )
        r = int(np.floor(r_f / w_f))
        c = int(np.floor(c_f / w_f))

        # Slice (3, 2*EXT, 2*EXT) centered at (r, c). The raster is pre-padded
        # by EXT px of -1.0 during preprocessing, so the slice stays in bounds
        # as long as the agent is within the recording's content bbox; the
        # check below fails loud otherwise.
        H_full, W_full = self._semantic_map.size(-2), self._semantic_map.size(-1)
        if not (EXT <= r <= H_full - EXT and EXT <= c <= W_full - EXT):
            raise RuntimeError(
                f"Agent pivot ({x0}, {y0}) maps to ({r}, {c}) which is too "
                f"close to the raster boundary (raster {H_full}x{W_full}, "
                f"need EXT={EXT} margin). Recording-specific issue — check "
                "the orthoPxToMeter and origin used in pixel_metric_shim.py."
            )

        # Crop: (3, 2*EXT, 2*EXT) = (3, 404, 404), on-device float32.
        patch = self._semantic_map[
            :, r - EXT : r + EXT, c - EXT : c + EXT
        ]

        # --- Stage 2: affine_grid rotation + final crop --------------------
        # (1, 2, 3) rotation for affine_grid, using the same sign convention
        # as the loader (angle = -heading). affine_grid maps OUTPUT pixels
        # back to INPUT sample locations, i.e. it is the inverse of rotating
        # the image, which is why this form (not its transpose) is correct.
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        rot = torch.tensor(
            [[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]],
            dtype=torch.float32,
            device=self.device,
        )  # (1, 2, 3)

        # affine_grid/grid_sample need a leading batch dim.
        m = patch.unsqueeze(0)
        grid = torch.nn.functional.affine_grid(
            rot, m.size(), align_corners=False
        )  # (1, 2*EXT, 2*EXT, 2)
        m = torch.nn.functional.grid_sample(
            m, grid, align_corners=False
        )  # (1, 3, 2*EXT, 2*EXT)
        # Asymmetric slice to (1, 3, MAP_SIZE, MAP_SIZE): more pixels ahead of
        # the agent than behind.
        m = m[..., _CROP_TOP:_CROP_BOTTOM, _CROP_LEFT:_CROP_RIGHT]

        # Add the leading 1 the encoder expects before it reshapes
        # (B, N, 3, H, W) -> (B*N, 3, H, W). N=1 at inference.
        return m.unsqueeze(0)  # (1, 1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Smoke test: synthetic input, no server.
# Run via:  python uav_guidance/server_code_only/main_program/contextvae_inference.py
#
# Drives the inferencer with a hand-crafted target moving diagonally and two
# parallel neighbors, and checks: output shape (6, 2), no NaN/Inf, and
# plausible per-step displacement (< 5 m at 5 Hz).
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Smoke-test the inferencer against a synthetic single-target scene."""
    # Use a real inD_01 val-split map/checkpoint so the crop geometry is valid.
    ckpt = os.path.join(_REPO_ROOT, "tmp", "levelx_full_map_v1", "ckpt-best")
    cfg = os.path.join(_REPO_ROOT, "configs", "levelx_train.py")
    map_pickle = os.path.join(_REPO_ROOT, "data", "levelx", "map", "inD_01.pkl")

    print(f"[smoke] loading ckpt   : {ckpt}")
    print(f"[smoke] loading config : {cfg}")
    print(f"[smoke] loading map    : {map_pickle}")

    inf = ContextVAEInferencer(
        ckpt_path=ckpt,
        config_path=cfg,
        map_pickle_path=map_pickle,
        device=None,  # auto-pick cuda if available
        # Pin the architecture to the M-ATTN checkpoint regardless of the
        # local config's current map_model; override to None for a no-map ckpt.
        model_overrides={"map_model": "resnet18"},
    )
    print(
        f"[smoke] loaded ckpt epoch={inf._loaded_epoch} "
        f"ADE={inf._loaded_ade:.4f} FDE={inf._loaded_fde:.4f}"
    )
    print(f"[smoke] device         : {inf.device}")

    # Synthetic target: 10 frames moving diagonally at ~1 m per timestep. The
    # pivot must map onto a valid raster pixel via H, so we read a real target's
    # first-obs-frame position from the val split rather than invent coords.
    val_txt = os.path.join(_REPO_ROOT, "data", "levelx", "val", "inD_01.txt")
    pivot = None
    with open(val_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Schema: fid aid x y heading group
            if len(parts) >= 4:
                try:
                    pivot = (float(parts[2]), float(parts[3]))
                    break
                except ValueError:
                    continue
    if pivot is None:
        raise RuntimeError(
            f"Could not find a sample row in {val_txt} to use as the smoke-test pivot."
        )
    print(f"[smoke] pivot (x, y)   : {pivot}")

    # Target moves 1.0 m along +x per timestep starting from the pivot.
    target_hist = np.array(
        [[pivot[0] + 1.0 * t, pivot[1] + 0.5 * t] for t in range(OB_HORIZON)],
        dtype=np.float64,
    )

    # Two neighbors parallel to the target, offset +/-3 m laterally.
    neighbors = {
        101: target_hist + np.array([0.0, 3.0])[None, :],
        102: target_hist + np.array([0.0, -3.0])[None, :],
    }

    # Supply heading (0 rad = +x) to exercise the heading-aware path and avoid
    # the one-shot fallback warning.
    out = inf.infer(target_hist, neighbors, heading_world=0.0)
    print(f"[smoke] output shape   : {out.shape}")
    print(f"[smoke] output[:3]     :\n{out[:3]}")
    assert out.shape == (PROTOCOL_FUTURE_STEPS, 2), f"unexpected shape {out.shape}"
    assert np.isfinite(out).all(), "output contains NaN or Inf"
    # Step displacement should be plausible — under a few meters at 5 Hz.
    step_disp = np.linalg.norm(np.diff(out, axis=0), axis=-1)
    print(f"[smoke] per-step disp  : {step_disp}")
    assert (step_disp < 5.0).all(), (
        f"per-step displacement {step_disp} exceeds 5 m — model is probably "
        "predicting nonsense; check input scale and map registration."
    )

    # Also exercise the K-samples entry point used by replay_video.py.
    samples = inf.infer_samples(target_hist, neighbors, heading_world=0.0)
    K_expected = inf.k_samples
    print(f"[smoke] samples shape  : {samples.shape}")
    assert samples.shape == (K_expected, PRED_HORIZON, 2), (
        f"unexpected samples shape {samples.shape}; "
        f"expected ({K_expected}, {PRED_HORIZON}, 2)"
    )
    assert np.isfinite(samples).all(), "samples contain NaN or Inf"
    # The prior draws are stochastic, so don't compare against infer()'s mean;
    # just check the K endpoints spread out (modes didn't fully collapse).
    endpoints = samples[:, -1, :]  # (K, 2)
    spread = float(np.linalg.norm(endpoints.std(axis=0)))
    print(f"[smoke] endpoint spread: {spread:.4f} m (>=0 ok; ~0 means modes collapsed)")
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
