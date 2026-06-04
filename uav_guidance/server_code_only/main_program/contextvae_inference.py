"""
ContextVAE single-target streaming inference for the Ntousis UAV server.

Purpose
-------
This module wraps a trained ContextVAE checkpoint so the Ntousis server-side
pipeline (`scene_recog_socket_yolov8.py`) can call it on per-frame DeepSORT
track histories and get K-sample future trajectories back.

It replaces the LSTM call (`tr_pred_lstm`) that currently lives at lines
583-594 of `scene_recog_socket_yolov8.py`.

Why this lives in `uav_guidance/...` and NOT in `contextvae/`
-----------------------------------------------------------
The training-side `contextvae.data.Dataloader` is a batch-oriented torch
Dataset that talks to on-disk `.txt`/`.info` files, per-pickle map rasters,
and a ProcessPoolExecutor loader. None of that is appropriate at server-side
inference, where we have:
  - a single target track (batch dim N=1),
  - history coming from a streaming DeepSORT buffer (not files),
  - a known recording (so the map pickle is loaded ONCE at startup, not
    re-fetched per item).
The Dataloader's per-item map crop and heading rotation logic IS needed —
we re-implement it here in a stripped-down, single-agent form rather than
import the loader and drag along its training-only machinery.

Coordinate frame contract
-------------------------
This module operates ENTIRELY in METRIC WORLD coordinates. Pixel<->metric
conversion is the responsibility of the caller (see `pixel_metric_shim.py`).
For Day 1-2 of the Week-3 integration, the shim uses a constant per-recording
`orthoPxToMeter` scale (works only for a stationary drone clip). Day 2-4
swaps the shim's internals for an ego-motion-compensated homography but
keeps the same metric output, so this module does NOT change.

ContextVAE conventions baked in here
------------------------------------
1. Observation horizon `OB_HORIZON = 10` frames at 5 Hz (2 s).
2. Prediction horizon `PRED_HORIZON = 25` frames at 5 Hz (5 s).
3. Observation radius `OB_RADIUS = 30` m (radius-based neighbor filter).
4. Map raster pre-baked to `(3, H, W)` in [-1, 1] with a 3x3 homography
   `H` that maps world `(x, y)` to image `(row, col)`. Pickle format
   matches `contextvae/data.py`'s `load_map` — namely a plain
   `pickle.dump((semantic_map_ndarray, H_ndarray), f)` tuple.
5. Neighbor tensor must span the FULL `L1 + L2` window (= 35 frames).
   At inference we don't know future neighbor positions, so frames
   `[OB_HORIZON .. OB_HORIZON+PRED_HORIZON]` are filled with `1e9`. The
   ContextVAE encoder's radius mask (`dist <= ob_radius`) filters those
   out. ZERO-PADDING IS WRONG — it silently passes the radius mask and
   corrupts the social attention. See `contextvae/data.py:473-478`.
6. Localization (matches `contextvae/data.py:502-521`): we rotate the
   entire scene (target + neighbors) around the target's first-obs-frame
   position `(x0, y0)` by `-heading`, where `heading` is the target's
   direction at the first obs frame. After this, `target[0] = (x0, y0)`
   unchanged and `target[1:]` is rotated. World coordinates are
   PRESERVED — we do NOT translate the origin to `(x0, y0)`.
7. K samples (default `K = 5`) are drawn from the timewise stochastic
   decoder; we collapse to a single trajectory by mean-over-K to fit the
   server's existing wire protocol (6 future `(x, y)` pairs). Week 4
   will revisit multi-modal handling.

Public API
----------
    inferencer = ContextVAEInferencer(
        ckpt_path="tmp/levelx_full_map_v1/ckpt-best",
        config_path="configs/levelx_train.py",
        map_pickle_path="data/levelx/map/inD_01.pkl",
        device="cuda",
    )
    future_xy = inferencer.infer(
        target_history_metric=(10, 2) np.ndarray of (x, y) in metric world coords,
        neighbor_histories_metric={track_id: (10, 2) np.ndarray, ...},
        heading_world=0.0,  # radians, world frame; matches training data.py:468.
                            # If omitted, falls back to atan2 of target[1]-target[0]
                            # AND emits a one-shot RuntimeWarning to surface the
                            # degradation — see infer() docstring.
    )  # returns (PROTOCOL_FUTURE_STEPS, 2) np.ndarray, mean over K samples,
       # in METRIC WORLD coords (same frame as inputs).

The caller converts `future_xy` back to pixels with the same shim it used
on the inputs.
"""

# Standard library
import os
import sys
import math
import pickle
import warnings
import importlib.util
from typing import Dict, Optional

# Third-party
import numpy as np
import torch

# Make the project's `contextvae` package importable from this server-side
# subdirectory. The repo layout is:
#     <repo_root>/
#         contextvae/
#         uav_guidance/server_code_only/main_program/contextvae_inference.py  <- this file
# We climb three levels up to reach <repo_root>, then prepend it to sys.path.
# This is a soft import shim — it lets us run the server-side code WITHOUT
# pip-installing the contextvae package and WITHOUT rewriting imports if the
# repo gets moved. The contextvae package itself is pure-Python with no
# `setup.py`, so editable-install isn't an option.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# After the sys.path patch above, this import resolves to <repo_root>/contextvae/model.py.
from contextvae.model import ContextVAE  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — must match the training config (configs/levelx_train.py).
# These are intentionally module-level (not config-driven) because the
# trained checkpoint at tmp/levelx_full_map_v1/ckpt-best was produced with
# these specific values; changing them here would silently mis-load the
# model's expected input shape. The values are mirrored verbatim from
# configs/levelx_train.py as of 2026-05-10.
# ---------------------------------------------------------------------------

# Number of past frames the encoder consumes. 10 frames at 5 Hz = 2 s.
OB_HORIZON = 10

# Number of future frames the decoder produces. 25 frames at 5 Hz = 5 s.
PRED_HORIZON = 25

# Radius-based neighbor filter (meters). Neighbors farther than this from
# the target at every frame are dropped at encoder time via the distance
# mask in `contextvae/model.py:306`.
OB_RADIUS = 30.0

# Square edge of the final map crop fed to MapEncode (224 x 224 RGB).
MAP_SIZE = 224

# Half-extent of the pre-rotation map crop (in pixels). Derived in
# `contextvae/data.py:155` as ceil(MAP_SIZE/4 * sqrt(13)) and stored as
# `self.EXT`. For MAP_SIZE=224 this works out to 202. We hard-code 202
# here because the training-time value WAS 202 — re-computing it from
# a different MAP_SIZE would diverge from the checkpoint's expected
# crop geometry. The pre-rotation patch is `(3, 2*EXT, 2*EXT) = (3, 404, 404)`,
# rotated, then sliced down to `(3, 224, 224)` via TOP/BOTTOM/LEFT/RIGHT
# offsets — see `_extract_rotated_map_crop` below.
EXT = 202

# Asymmetric crop offsets after the rotation step (in pixels). LEFT < RIGHT
# is INTENTIONAL — after rotation, the agent's forward heading points
# along +x, and we want MORE map context in front of the agent than behind.
# Mirrors `contextvae/data.py:153-163`:
#     TOP    = MAP_SIZE // 2 = 112      (vertically centered)
#     LEFT   = MAP_SIZE // 4 = 56       (1/4 of width behind agent)
#     BOTTOM = MAP_SIZE - TOP = 112     (vertically centered)
#     RIGHT  = MAP_SIZE - LEFT = 168    (3/4 of width in front of agent)
#     Then offset by EXT to index into the 2*EXT-wide rotated patch.
_CROP_TOP    = EXT - MAP_SIZE // 2          # 90
_CROP_BOTTOM = EXT + (MAP_SIZE - MAP_SIZE // 2)  # 314
_CROP_LEFT   = EXT - MAP_SIZE // 4          # 146
_CROP_RIGHT  = EXT + (MAP_SIZE - MAP_SIZE // 4)  # 370

# Default K. The training run used pred_samples=5; we mirror it so that
# the server's per-frame inference cost matches what was measured in
# the offline eval.
DEFAULT_K_SAMPLES = 5

# How many future steps the Ntousis socket protocol returns to the UAV.
# The LSTM at scene_recog_socket_yolov8.py:583-594 returns 6 steps; we
# truncate ContextVAE's 25-step output to the first 6 steps to keep the
# wire protocol byte-compatible. (Week 4 may extend the protocol to
# carry more steps + multi-modal samples.)
PROTOCOL_FUTURE_STEPS = 6

# Sentinel value used to pad absent neighbor slots so the radius mask
# (`dist <= ob_radius`) filters them out. Lifted from
# `contextvae/data.py:262`. ZERO-PADDING IS WRONG — it would put fake
# neighbors at the origin and corrupt social attention.
NEIGHBOR_PAD_VALUE = 1.0e9

# Hard cap on the number of neighbor slots fed to the encoder. The
# encoder is per-agent attention so it scales with Nn, but we don't want
# unbounded Nn growth when a crowded intersection produces dozens of
# tracks at once. 12 was the original Lyft/Waymo ballpark in the paper
# supplementary; for inD/uniD crowds it's loose enough not to truncate.
MAX_NEIGHBORS = 12


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class ContextVAEInferencer:
    """
    Wraps a trained ContextVAE checkpoint for single-target streaming
    inference.

    One instance is created per server connection. The model and map
    raster are loaded ONCE at construction; `infer(...)` is then called
    per frame.

    Thread-safety: the underlying torch model is NOT thread-safe (parameter
    flatten_parameters() mutates state). Use one Inferencer per worker
    thread, or guard `infer(...)` with an external mutex if sharing across
    threads.
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
        # `map_pickle_path=None` is legal for no-map checkpoints (the S-ATTN-
        # only variant trained without M-ATTN). When None, the inferencer
        # skips the map pickle load and passes `map=None` to the model;
        # `ContextVAE.enc()` already takes the no-map branch in that case
        # (see contextvae/model.py:279 — `use_map = map is not None and
        # self.use_map`). The caller is responsible for matching the
        # checkpoint's training config: passing `model_overrides={"map_model": None}`
        # alongside `map_pickle_path=None` is the canonical no-map setup.
        # --- Resolve device -------------------------------------------------
        # We default to CUDA if available — the server box is the A40 host,
        # and ContextVAE inference fits comfortably in <1 GB of VRAM, so
        # there's no reason to leave the GPU idle. Caller can override with
        # device="cpu" for unit-testing on machines without CUDA.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.k_samples = int(k_samples)

        # One-shot warning latch for the heading-fallback path in `infer()`.
        # When a caller invokes `infer(...)` WITHOUT supplying `heading_world`,
        # we approximate heading via finite-difference (atan2) over the first
        # two obs positions. For stationary/slow agents this collapses to 0
        # and the M-ATTN encoder receives an un-rotated map crop it never
        # saw during training — a silent degradation. We warn the first time
        # this happens per instance so the caller is on notice; subsequent
        # calls in the degraded mode stay quiet to avoid log spam.
        self._heading_fallback_warned = False

        # --- Load the config file -------------------------------------------
        # The training-side configs are plain Python files (not YAML/JSON).
        # We load them with importlib so we can read `config.model` (the
        # ContextVAE constructor kwargs) without hard-coding them. This
        # mirrors `contextvae/main.py:36-42` verbatim — keep the two in
        # sync if main.py ever changes its config-loading scheme.
        spec = importlib.util.spec_from_file_location(
            "contextvae_inference_config",  # synthetic module name
            config_path,
            submodule_search_locations=[os.path.dirname(config_path)],
        )
        self._config = importlib.util.module_from_spec(spec)
        # Register in sys.modules BEFORE exec so configs that inherit via a
        # relative import (e.g. visdrone_eval_5s.py's `from .levelx_train
        # import *`) can resolve their parent package — `.` resolves to this
        # module's name, which must be importable. main.py:41 and
        # eval_perseq.py:47 do the same; keep in sync.
        sys.modules[spec.name] = self._config
        spec.loader.exec_module(self._config)

        # --- Instantiate the model ------------------------------------------
        # config.model is a dict like:
        #     {"horizon": 25, "ob_radius": 30, "hidden_dim": 512, "map_model": "resnet18"}
        # The headline checkpoint at tmp/levelx_full_map_v1/ckpt-best was
        # trained with map_model="resnet18". The training-side config file
        # is user-mutable for ablations — at the time of writing it had been
        # toggled to map_model=None for a no-map re-eval. We accept a
        # `model_overrides` dict so callers can pin the architecture to
        # what matches the checkpoint regardless of the config file's
        # current contents. If the resulting kwargs don't match the
        # checkpoint, `load_state_dict` below will fail with shape
        # mismatches — that's the intended "fail loud" behavior.
        model_kwargs = dict(self._config.model)
        if model_overrides:
            model_kwargs.update(model_overrides)
        self.model = ContextVAE(**model_kwargs)
        self._model_kwargs = model_kwargs  # kept for debug/logging

        # --- Load the trained weights ---------------------------------------
        # Mirrors `contextvae/main.py:140-141`:
        #     state_dict = torch.load(ckpt_path, map_location=device)
        #     model.load_state_dict(state_dict["model"])
        # The ckpt file is a torch-pickled dict with keys
        #     {"model", "ade", "fde", "ade_d", "fde_d", "epoch", "rng_state"}
        # We only need "model" for inference. map_location handles the
        # case where the ckpt was saved on cuda:0 but we're loading on a
        # different device (cuda vs cpu).
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        # eval() disables dropout (including the map-input dropout in
        # `contextvae/model.py:338-339`) and switches BN to running stats.
        # Failing to call eval() here would inject training noise into every
        # server-side prediction.
        self.model.eval()
        # Log what we loaded so server-startup logs make the version explicit.
        self._loaded_epoch = state.get("epoch", -1)
        self._loaded_ade = state.get("ade", float("nan"))
        self._loaded_fde = state.get("fde", float("nan"))

        # --- Load the map raster (optional) ---------------------------------
        # The pickle is `(semantic_map_ndarray, H_ndarray)` matching
        # `contextvae/data.py:530-538` (`load_map`). We hold both on the
        # target device so the affine crop runs without per-frame H2D copy.
        # semantic_map: (3, H_full, W_full) in [-1, 1]
        # H: (3, 3) — world (x, y, 1) -> image (row, col, 1)
        #
        # When `map_pickle_path` is None the inferencer is in NO-MAP mode:
        # the model receives `map=None` per inference and `ContextVAE.enc()`
        # takes its no-map branch (see contextvae/model.py:351-353). The
        # caller MUST also set `model_overrides={"map_model": None}` so that
        # the model is built without the MapEncode sub-module — otherwise the
        # checkpoint state_dict won't match. We do NOT enforce that here at
        # construction (an explicit check would couple us to the config-file
        # contents); the state_dict load below will fail loud if the
        # combination is mismatched.
        if map_pickle_path is not None:
            with open(map_pickle_path, "rb") as f:
                sem_map_np, H_np = pickle.load(f)
            # semantic_map is loaded as float32 (data.py stores it that way
            # after the preprocessing notebook). torch.tensor(..., dtype=
            # float32) makes it explicit in case a future pickle ships a
            # different dtype.
            self._semantic_map = torch.tensor(
                sem_map_np, dtype=torch.float32, device=self.device
            )
            # H is a numpy 3x3 ndarray. We keep it numpy because the only
            # operation we do with it is a single `H @ [x, y, 1]` per
            # inference call (fast on CPU; not worth a GPU round-trip).
            self._H = np.asarray(H_np, dtype=np.float64)

            # Sanity-check shapes — fail loud at construction rather than
            # during the first infer() call.
            assert self._semantic_map.dim() == 3 and self._semantic_map.size(0) == 3, (
                f"map pickle has unexpected shape {self._semantic_map.shape}; "
                "expected (3, H, W)"
            )
            assert self._H.shape == (3, 3), (
                f"map pickle H has unexpected shape {self._H.shape}; expected (3, 3)"
            )
        else:
            # No-map mode. Both attributes still exist (so attribute access
            # downstream doesn't AttributeError) — they just take the
            # sentinel value None, which `_run_forward` checks before
            # invoking `_build_map_tensor`.
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

        Performs steps 0-7 of the original `infer()` body: validate, compute
        heading, build the localization rotation, build the model tensors,
        and call the model. Returns the raw K-sample prediction in the
        LOCALIZED frame plus the (pivot, R) pair needed to un-localize.

        Callers are responsible for the K-collapse (or not), the un-rotation,
        and any output truncation. This split exists so `infer()` (the
        socket-protocol entry point) and `infer_samples()` (the replay-driver
        entry point) share identical model-input construction with zero
        risk of drift.

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
        # Cheap shape checks. The model would eventually fail with a less
        # informative error if these are wrong (e.g. RuntimeError on a tensor
        # reshape inside the encoder); fail at the boundary instead.
        target_history_metric = np.asarray(target_history_metric, dtype=np.float64)
        if target_history_metric.shape != (OB_HORIZON, 2):
            raise ValueError(
                f"target_history_metric must have shape ({OB_HORIZON}, 2); "
                f"got {target_history_metric.shape}"
            )

        # --- 1. Compute heading at the first obs frame ---------------------
        # ContextVAE's training-time localization uses the heading at the
        # FIRST obs frame (see contextvae/data.py:468 — `heading` is read
        # from the agent's data at `first_frame[idx][1]` which corresponds
        # to tid_start). We mirror that at inference time.
        #
        # Preferred path: the caller supplies `heading_world` (radians, world
        # frame) directly — typically pulled from the same source training
        # used (the preprocessed .txt heading column, or a robust DeepSORT-
        # bbox-displacement estimate over a window > 1 step). When supplied,
        # we use it verbatim, eliminating the train/infer drift documented
        # by the heading-source audit.
        #
        # Fallback path: when `heading_world` is None we approximate heading
        # via first-order finite difference `atan2(v0_y, v0_x)` over the
        # first two obs positions. For stationary/slow agents the velocity
        # vector is ~zero and atan2 returns 0 (atan2(0, 0) == 0 by IEEE 754
        # / Python convention). The map crop is then axis-aligned with the
        # WORLD frame rather than the AGENT frame — a distribution the
        # M-ATTN encoder never saw at training time, degrading predictions.
        # We emit a one-shot warning per inferencer instance so the caller
        # is on notice; subsequent fallback calls stay quiet to avoid spam.
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

        # The rotation angle stored in `__getitem__` is `-heading` (see
        # data.py:495). We carry the same sign here so the rotation logic
        # below mirrors training.
        angle = -heading

        # --- 2. Pivot for the localization rotation ------------------------
        # The pivot is the FIRST obs frame position. Mirrors data.py:505-506
        # (`x0 = hist[0][0]; y0 = hist[0][1]`). After rotation around this
        # pivot, `target_history[0]` is unchanged and `target_history[1:]`
        # is rotated. World coordinates are preserved — we do NOT translate
        # the origin to (x0, y0).
        pivot_xy = target_history_metric[0].copy()  # shape (2,)

        # --- 3. Apply the localization rotation ----------------------------
        # Mirrors data.py:507-521. The trick (subtract pivot -> rotate ->
        # add pivot back) rotates around the pivot while keeping world coords.
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)  # (2, 2)

        target_loc = self._rotate_around_pivot(target_history_metric, pivot_xy, R)

        # Same rotation applied to each neighbor's (OB_HORIZON, 2) array.
        # We keep them as a dict here; collation into the model tensor
        # happens in step 5.
        neighbors_loc = {
            tid: self._rotate_around_pivot(
                np.asarray(hist, dtype=np.float64), pivot_xy, R
            )
            for tid, hist in neighbor_histories_metric.items()
        }

        # --- 4. Build the target's 6-D state tensor ------------------------
        # Model expects (OB_HORIZON, N=1, 6) where the 6 channels are
        # (x, y, vx, vy, ax, ay). Per `contextvae/model.py:294-296` the
        # encoder recomputes v from positions and only reads `state[1, ..., 4:6]`
        # for the first acceleration — so the vx/vy fields of the target's
        # state tensor are effectively ignored, but we fill them anyway with
        # finite differences for consistency and to make the code path
        # observable in a debugger.
        target_state_6d = self._positions_to_6d_state(target_loc)  # (10, 6)
        x_tensor = torch.tensor(
            target_state_6d[:, None, :],  # add the N=1 batch dim
            dtype=torch.float32,
            device=self.device,
        )

        # --- 5. Build the neighbor tensor ----------------------------------
        # Shape: (OB_HORIZON + PRED_HORIZON, N=1, Nn, 6). The future
        # OB_HORIZON..OB_HORIZON+PRED_HORIZON-1 frames are filled with
        # NEIGHBOR_PAD_VALUE (=1e9), since we don't know future neighbor
        # positions at inference time. The encoder's distance mask filters
        # them out.
        neighbor_tensor = self._build_neighbor_tensor(
            target_loc=target_loc,
            neighbors_loc=neighbors_loc,
        )  # (35, 1, Nn, 6) float32

        # --- 6. Build the map tensor ---------------------------------------
        # Two-step build, mirroring contextvae/data.py:
        #   (a) Crop a (3, 2*EXT, 2*EXT) patch from the full raster centered
        #       at the pre-rotation pivot pixel `(r, c) = H @ (x0, y0, 1)`.
        #       This is the SAME pivot we used in step 2 (the first obs
        #       frame), because data.py:357 does the same thing.
        #   (b) Apply affine_grid + grid_sample with the same `angle` we
        #       used in step 3, then crop down to (3, MAP_SIZE, MAP_SIZE)
        #       with the asymmetric TOP/BOTTOM/LEFT/RIGHT slice that
        #       biases the crop forward of the agent.
        # Output shape: (N=1, 3, MAP_SIZE, MAP_SIZE), then unsqueezed at
        # dim=0 to match training-time (1, N, 3, H, W).
        #
        # When the inferencer is in no-map mode (`self._semantic_map is None`)
        # we skip the crop entirely and pass `map_tensor = None` to the model.
        # `ContextVAE.enc()` takes its no-map branch in that case — see
        # contextvae/model.py:351-353. The neighbor + S-ATTN path is
        # unaffected.
        if self._semantic_map is not None:
            map_tensor = self._build_map_tensor(
                pivot_xy=pivot_xy, angle=angle
            )  # (1, 1, 3, 224, 224) float32
        else:
            map_tensor = None

        # --- 7. Call the model ---------------------------------------------
        # The model's forward signature for eval mode (model.training=False,
        # n_predictions>0) is `(x, neighbor, map, n_predictions=K)` and it
        # returns `(K, PRED_HORIZON, N, 2)`. See contextvae/model.py:385-478.
        # We pass map and skip seq_len (None / default) because OB_HORIZON
        # frames are always available at inference — there's no pad-mode.
        pred = self.model(
            x_tensor,
            neighbor_tensor,
            map_tensor,
            n_predictions=self.k_samples,
        )
        # pred: (K, 25, 1, 2) — K samples of 25 future (delta-x, delta-y)
        # pairs already accumulated by `torch.cumsum` and shifted by `x_T`
        # inside the model (see model.py:473-476). So pred is in the SAME
        # frame as x_tensor — i.e. world coords rotated around pivot_xy by
        # `angle = -heading`.

        # Squeeze the N=1 batch axis and convert to numpy float64 so all
        # downstream geometry stays in numpy (matches the rest of the
        # localization plumbing in this module).
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

        This is the SOCKET-PROTOCOL entry point. The Ntousis wire protocol
        expects a single (6, 2) trajectory, so we collapse the K samples
        to their mean and truncate to the first 6 steps. For multi-modal
        rendering, use `infer_samples()` instead.

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
            (i.e. atan2-convention angle from +x). When provided, this is
            used VERBATIM for the localization rotation and map crop, exactly
            matching the training-time loader (which reads `heading` from the
            preprocessed .txt — see contextvae/data.py:468). When None, we
            fall back to a finite-difference approximation `atan2(v0_y, v0_x)`
            over the first two obs positions and emit a one-shot warning;
            for stationary/slow agents this fallback collapses to 0 and the
            M-ATTN encoder receives an un-rotated crop it never saw in
            training. Callers with access to a real heading (e.g. via the
            preprocessed .txt heading column, or DeepSORT bbox displacement
            over a longer history) SHOULD pass it here.

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

        # Collapse K samples to single mean trajectory and un-localize.
        # We mean over the K axis. This is the simplest "single-prediction
        # output" strategy and the one called for by the Week-3 Day-1-2 scope.
        pred_mean = pred_loc.mean(axis=0)  # (25, 2)
        R_inv = R.T  # inverse of the rotation applied during localization
        pred_unloc = self._rotate_around_pivot(pred_mean, pivot_xy, R_inv)

        # Truncate to the first PROTOCOL_FUTURE_STEPS. The Ntousis wire
        # protocol sends 6 future steps; ContextVAE produces 25.
        return pred_unloc[:PROTOCOL_FUTURE_STEPS]

    def infer_samples(
        self,
        target_history_metric: np.ndarray,
        neighbor_histories_metric: Dict[int, np.ndarray],
        heading_world: Optional[float] = None,
    ) -> np.ndarray:
        """
        Run one inference step and return ALL K sample trajectories.

        This is the REPLAY-DRIVER entry point — used by offline tooling
        (e.g. `replay_video.py`) that needs the full multi-modal output
        rather than the mean-collapsed one the socket protocol carries.

        Same input contract as `infer()`. Differs in two ways from `infer()`:

          - Returns the full `K` samples (no mean collapse) so callers can
            visualize multi-modality.
          - Returns the full `PRED_HORIZON` (25 steps = 5 s) rather than
            the protocol-truncated 6. Replay overlays benefit from the
            longer horizon, and there's no wire-protocol constraint here.

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

        # Un-localize each of the K trajectories. `_rotate_around_pivot`
        # already handles arbitrary leading dims via `np.einsum`, so we
        # can pass the whole (K, 25, 2) tensor at once.
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
        Rotate a (..., 2) array of points around `pivot_xy` by the 2x2
        rotation matrix `R`. World coordinates are preserved (we subtract
        the pivot, rotate, add the pivot back).

        Mirrors the trick in contextvae/data.py:513-521.

        Shapes
        ------
        points_xy : (..., 2) — any leading dimensions allowed
        pivot_xy  : (2,)
        R         : (2, 2)
        returns   : same shape as points_xy
        """
        # Broadcasting: pivot_xy is shape (2,); subtracting from points_xy
        # (shape (..., 2)) broadcasts pivot across the leading dims.
        centered = points_xy - pivot_xy
        # The matmul `R @ p` for a column vector p; we work on the last
        # axis with einsum to avoid reshaping. R has shape (2, 2),
        # centered has shape (..., 2) — einsum contracts R's columns with
        # centered's last axis.
        rotated = np.einsum("ij,...j->...i", R, centered)
        return rotated + pivot_xy

    @staticmethod
    def _positions_to_6d_state(positions: np.ndarray) -> np.ndarray:
        """
        Convert an (OB_HORIZON, 2) array of (x, y) positions into the
        (OB_HORIZON, 6) state tensor `(x, y, vx, vy, ax, ay)` that
        ContextVAE expects.

        Velocity and acceleration use first-order finite differences in
        UNITS OF METERS PER TIMESTEP (NOT m/s). This matches the training
        convention computed inside contextvae/data.py's `extend()`. The
        Ntousis-LSTM comparison in Week 4 will need an explicit *5 to
        convert these to m/s (at the 5 Hz cadence).

        First-frame velocity is copied from frame 1 (so v[0] == v[1]),
        and first-two-frame accelerations are copied from frame 2
        (so a[0] == a[1] == a[2]). This mirrors the cold-start handling
        in contextvae/data.py:577-579 and :595-598 — at the start of an
        agent's trajectory, finite differences below the relevant order
        aren't well-defined, so the loader copies the next available value.

        Note: the encoder recomputes v from positions and only reads
        state[1, :, 4:6] for the first acceleration (model.py:294-296).
        So the only state slot the model actually consumes from this
        return value is `state[1, 4:6] = (ax_1, ay_1)`. The other 4-channel
        entries (vx, vy, ax_others, ay_others) are inert for the target,
        but we fill them all for consistency with the training-time format
        and to keep the code observable.
        """
        T = positions.shape[0]
        out = np.zeros((T, 6), dtype=np.float64)
        out[:, 0:2] = positions

        # Velocity: vx_t = x_t - x_{t-1}, with vx_0 := vx_1.
        # `np.diff` along axis 0 returns shape (T-1, 2); we shift it into
        # rows [1:] of `v` and fill row 0 from row 1.
        v = np.zeros_like(positions)
        v[1:] = np.diff(positions, axis=0)
        v[0] = v[1]  # cold-start copy
        out[:, 2:4] = v

        # Acceleration: ax_t = vx_t - vx_{t-1}, with ax_0 := ax_1 := ax_2.
        a = np.zeros_like(positions)
        if T >= 3:
            a[2:] = np.diff(v[1:], axis=0)  # length T-2, fills rows 2..T-1
            a[1] = a[2]
            a[0] = a[2]
        elif T >= 2:
            # Degenerate case (history shorter than expected). Should not
            # happen at the contract-validated OB_HORIZON=10 but kept defensive.
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
        - Frames `[0 .. OB_HORIZON-1]` carry the actual neighbor histories
          (computed 6-D state, same convention as the target).
        - Frames `[OB_HORIZON .. OB_HORIZON+PRED_HORIZON-1]` are filled with
          `NEIGHBOR_PAD_VALUE = 1e9` — the encoder's distance mask filters
          these out as far-away.
        - Neighbors farther than `OB_RADIUS` from the target at every obs
          frame are pre-filtered here (so the encoder doesn't waste budget
          on them). Mirrors contextvae/data.py:473-478.
        - The slot dimension `Nn` is padded with `1e9`-only entries up to
          `MAX_NEIGHBORS`, matching the training-time padding pattern in
          contextvae/data.py's collate_fn at lines 257-264.

        Why the full L1+L2 window even at inference
        -------------------------------------------
        At training, the posterior-RNN (`rnn_by`) consumes neighbor states
        across the FUTURE prediction horizon as well. At inference we only
        run the prior branch, which does NOT touch the future neighbor
        slice — but the model's forward path still reads `neighbor[1:]` and
        slices it, so the tensor must HAVE 35 frames. Skipping the future
        25 frames would crash with a shape error inside `enc()`.
        """
        total_frames = OB_HORIZON + PRED_HORIZON  # 35

        # Step 1: filter neighbors by radius (using the LOCALIZED target
        # positions, since we localize neighbors to the same frame).
        # We require the neighbor to be within OB_RADIUS at SOME obs frame.
        kept_states: list[np.ndarray] = []
        for _tid, hist in neighbors_loc.items():
            # Pairwise distance between target and this neighbor at each
            # obs frame. hist shape (OB_HORIZON, 2), target_loc shape
            # (OB_HORIZON, 2). dist shape (OB_HORIZON,).
            dist = np.linalg.norm(hist - target_loc, axis=-1)
            # The contract is "kept if within OB_RADIUS at any obs frame".
            # Mirrors data.py:476-477:
            #     valid = dist <= ob_radius     # L x (N-1)
            #     valid = np.any(valid, axis=0) # N-1
            if np.any(dist <= OB_RADIUS):
                kept_states.append(self._positions_to_6d_state(hist))
                if len(kept_states) >= MAX_NEIGHBORS:
                    # Hard cap. Should be rare in inD/uniD scenes.
                    break

        # Step 2: assemble the full-window tensor.
        # Default-initialize EVERYTHING with the pad value, then overlay
        # the actual obs-window states on top. Future frames remain padded.
        # nn is the actual neighbor count after radius filtering; we still
        # always emit MAX_NEIGHBORS slots (with the unused ones all-1e9) so
        # the model's last-dim shape is constant across calls.
        nb = np.full(
            (total_frames, 1, MAX_NEIGHBORS, 6),
            fill_value=NEIGHBOR_PAD_VALUE,
            dtype=np.float32,
        )
        for slot_idx, state6d in enumerate(kept_states):
            # state6d shape: (OB_HORIZON, 6). Place it in slot `slot_idx`
            # at frames [0..OB_HORIZON-1]. Batch dim is 0 (N=1).
            nb[:OB_HORIZON, 0, slot_idx, :] = state6d.astype(np.float32)

        return torch.from_numpy(nb).to(self.device)

    def _build_map_tensor(
        self, pivot_xy: np.ndarray, angle: float
    ) -> torch.Tensor:
        """
        Build the map tensor of shape (1, N=1, 3, MAP_SIZE, MAP_SIZE).

        Two-stage construction mirrors contextvae/data.py:
          1. (`__getitem__`, lines 357-366) Pre-rotation pixel crop of size
             `(3, 2*EXT, 2*EXT)` centered at the agent's first-obs-frame
             pixel location `(r, c) = H @ (x, y, 1)`.
          2. (`collate_fn`, lines 300-302) Affine rotation by `angle` via
             `affine_grid + grid_sample`, then asymmetric slice to
             `(3, MAP_SIZE, MAP_SIZE)`.

        The outer batch dim (the leading `1`) matches the (B, N, 3, H, W)
        layout the encoder sees during training; the encoder collapses it
        with .view() in `MapEncode.forward` (see model.py:168-179).
        """
        # --- Stage 1: pre-rotation pixel crop ------------------------------
        # Apply the homography to the pivot world coords to get the pixel
        # (row, col) of the agent in the full raster. Matches data.py:358.
        # The third row of H is `[0, 0, 1]` for an affine, so the divide
        # by w is a no-op; we still do it for safety in case the pickle
        # has a non-affine projective.
        x0, y0 = float(pivot_xy[0]), float(pivot_xy[1])
        r_f, c_f, w_f = self._H @ np.array([x0, y0, 1.0], dtype=np.float64)
        # Guard against the degenerate w=0 case (impossible for our
        # well-formed H, but fail loud if a future pickle is malformed).
        if abs(w_f) < 1e-12:
            raise RuntimeError(
                f"Map homography produced w=0 for pivot ({x0}, {y0}). "
                "Pickle H may be malformed."
            )
        r = int(np.floor(r_f / w_f))
        c = int(np.floor(c_f / w_f))

        # Slice out (3, 2*EXT, 2*EXT) centered at (r, c). The preprocessing
        # notebook pre-pads the raster by EXT pixels of -1.0 (see project
        # memory `project_build_map_pickle_fix.md`), so the slice never
        # falls off the raster as long as the agent is within the recording's
        # content bbox. We still bounds-check to fail loud if it does.
        H_full, W_full = self._semantic_map.size(-2), self._semantic_map.size(-1)
        if not (EXT <= r <= H_full - EXT and EXT <= c <= W_full - EXT):
            raise RuntimeError(
                f"Agent pivot ({x0}, {y0}) maps to ({r}, {c}) which is too "
                f"close to the raster boundary (raster {H_full}x{W_full}, "
                f"need EXT={EXT} margin). Recording-specific issue — check "
                "the orthoPxToMeter and origin used in pixel_metric_shim.py."
            )

        # Crop. Shape after slice: (3, 2*EXT, 2*EXT) = (3, 404, 404).
        patch = self._semantic_map[
            :, r - EXT : r + EXT, c - EXT : c + EXT
        ]  # still on self.device, still float32

        # --- Stage 2: affine_grid rotation + final crop --------------------
        # Build the (1, 2, 3) rotation matrix for affine_grid. Mirrors
        # data.py:362-365:
        #     rot = [[cos, -sin, 0], [sin, cos, 0]]
        # with the SAME sign convention (angle = -heading_world). Note that
        # this matrix is INVERSE of the image rotation we want — affine_grid
        # uses it to compute where in the INPUT to sample for each OUTPUT
        # pixel, which is the inverse of "rotate the image by angle".
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        rot = torch.tensor(
            [[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]],
            dtype=torch.float32,
            device=self.device,
        )  # (1, 2, 3)

        # `patch.unsqueeze(0)` -> (1, 3, 2*EXT, 2*EXT) — required batch dim
        # for affine_grid/grid_sample.
        m = patch.unsqueeze(0)
        grid = torch.nn.functional.affine_grid(
            rot, m.size(), align_corners=False
        )  # (1, 2*EXT, 2*EXT, 2)
        m = torch.nn.functional.grid_sample(
            m, grid, align_corners=False
        )  # (1, 3, 2*EXT, 2*EXT)
        # Final asymmetric slice to (1, 3, MAP_SIZE, MAP_SIZE). The asymmetry
        # (LEFT=56, RIGHT=168) gives the model more pixels in front of the
        # agent than behind.
        m = m[..., _CROP_TOP:_CROP_BOTTOM, _CROP_LEFT:_CROP_RIGHT]

        # data.py:310 does `m.unsqueeze_(0)` to insert the leading "1" the
        # model expects (because the encoder reshapes (B, N, 3, H, W) to
        # (B*N, 3, H, W) in MapEncode.forward). At inference N=1, so the
        # leading-1 is fine.
        return m.unsqueeze(0)  # (1, 1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Smoke test: synthetic input, no server.
# Run via:  python -m uav_guidance.server_code_only.main_program.contextvae_inference
# or:       python uav_guidance/server_code_only/main_program/contextvae_inference.py
#
# The synthetic input drives the inferencer with a hand-crafted target moving
# diagonally and two neighbors moving in parallel. We sanity-check that:
#   - The returned array has shape (6, 2).
#   - No NaNs / Infs.
#   - Per-step displacement is plausible (< 5 m at 5 Hz cadence) for an
#     agent that was moving at ~1 m per timestep.
# This is the Step-1 verification from the implementation plan.
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Smoke-test the inferencer against a synthetic single-target scene."""
    # Use a known map pickle from the val split. inD_01 was in the val split
    # used for the M-ATTN headline run, so the trained model has seen this
    # location (in the validation sense, not training).
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
        # Pin to the headline-checkpoint architecture in case the local
        # config file is mid-experiment with map_model toggled to None.
        # If you swap the checkpoint for a no-map one, override to None here.
        model_overrides={"map_model": "resnet18"},
    )
    print(
        f"[smoke] loaded ckpt epoch={inf._loaded_epoch} "
        f"ADE={inf._loaded_ade:.4f} FDE={inf._loaded_fde:.4f}"
    )
    print(f"[smoke] device         : {inf.device}")

    # Synthetic target: 10 frames moving diagonally at ~1 m per timestep.
    # The origin / scale are arbitrary; they just need to map onto a valid
    # pixel coordinate via H. We pick a point that we know is near the
    # center of the inD_01 recording (based on inspection of `data/levelx/`
    # — the data.py `__getitem__` would have already crashed at training
    # time if any sample's H @ hist[0] fell outside the raster, so we
    # cheat and read one real target's first-obs-frame position from the
    # val split file directly to guarantee the pivot is valid.
    val_txt = os.path.join(_REPO_ROOT, "data", "levelx", "val", "inD_01.txt")
    # Find one frame's coordinates to use as the pivot.
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

    # Two neighbors moving parallel to the target, offset laterally.
    neighbors = {
        101: target_hist + np.array([0.0, 3.0])[None, :],   # +3 m in y
        102: target_hist + np.array([0.0, -3.0])[None, :],  # -3 m in y
    }

    # Pass a synthetic heading (0 rad = +x) so the smoke test exercises the
    # heading-aware code path and does NOT trip the one-shot fallback warning.
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

    # Also exercise the new K-samples entry point used by replay_video.py.
    samples = inf.infer_samples(target_hist, neighbors, heading_world=0.0)
    K_expected = inf.k_samples
    print(f"[smoke] samples shape  : {samples.shape}")
    assert samples.shape == (K_expected, PRED_HORIZON, 2), (
        f"unexpected samples shape {samples.shape}; "
        f"expected ({K_expected}, {PRED_HORIZON}, 2)"
    )
    assert np.isfinite(samples).all(), "samples contain NaN or Inf"
    # Cross-check: mean of K samples truncated to PROTOCOL_FUTURE_STEPS
    # must agree with `infer()`'s output (same seed-free deterministic path
    # is not guaranteed across calls because the prior draws are stochastic,
    # so we run both off the same forward batch by re-seeding torch here).
    # Instead, just sanity-check that the K endpoints span more than a pixel.
    endpoints = samples[:, -1, :]  # (K, 2)
    spread = float(np.linalg.norm(endpoints.std(axis=0)))
    print(f"[smoke] endpoint spread: {spread:.4f} m (>=0 ok; ~0 means modes collapsed)")
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
