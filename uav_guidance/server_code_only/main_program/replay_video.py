"""
Offline replay driver: video in -> ContextVAE K=5 trajectory overlays out.

Purpose
-------
Renders the deliverable: an offline replay of a UAV video through
the integrated ContextVAE pipeline (`ContextVAEInferencer` +
`PixelMetricShim` / `MovingCameraShim`), producing K=5 future trajectory
samples per tracked agent as overlays on the source video.

Unlike `scene_recog_socket_yolov8.py` (the live socket server), this driver
runs entirely offline against a recorded mp4 (`--input`) OR a folder of
frame images (`--input-images`). The K=5 trajectories are kept individually
(not mean-collapsed) so the overlay shows the model's multi-modality.

Bbox sources (`--anno-format`)
------------------------------
Three ways to obtain the per-frame agent boxes the pipeline tracks:
  - `synth-json`    : ground-truth boxes from the synthetic-clip JSON
                      (`synth_clip_gen.py`). Used for the inD orthomap demo;
                      YOLOv8 can't reliably detect the rendered rectangles.
  - `visdrone-mot`  : ground-truth boxes from a VisDrone-MOT
                      `annotations.txt` (the GT-driven replay).
  - `yolo-deepsort` : live YOLOv8 + DeepSORT detection on the frames — the
                      "zero annotations" END-TO-END path. No annotation file;
                      pixel->metric scale is auto-calibrated from detected
                      car bboxes (override with `--scale-mpp`). Requires
                      `ultralytics` + `deep_sort_realtime` (present in the
                      `new_xupei_env` conda env).

Two-mode operation
------------------
- `--ego-motion=false` (stationary):
    Use `PixelMetricShim`. Pixel<->metric conversion assumes the live
    video frame == the un-padded `XX_background.png`. On the synthetic
    clip — which DOES pan/zoom — this should produce visibly drifting
    predictions. That drift is the failure mode the ego-motion shim
    addresses, and the side-by-side comparison is the qualitative
    artifact for Ch6.

- `--ego-motion=true` (moving camera):
    Use `MovingCameraShim`. Per-frame `update_frame(frame, masks)` runs
    ORB+RANSAC to estimate `H_live -> K0`, then composes with the static
    `H_K0 -> world` from the orthomap pickle. Predictions are world-frame
    anchored, so the overlay tracks the ground correctly even as the
    camera pans.

Inference cadence
-----------------
ContextVAE operates at 5 Hz. The synthetic clip is 25 fps with
`frame_repeat=5` (each underlying inD 5-Hz step is held for 5 video
frames). We therefore append to per-agent history once every `--fps-step`
video frames (default 5 -> 5 Hz history), and invoke `infer_samples()`
once per 5-Hz tick as soon as the chosen target has 10 history samples.
The most recent K=5 prediction is then rendered onto EVERY video frame
between ticks; because the trajectories are in world coords, the shim
converts them to live-pixel coords per frame, producing an overlay that
tracks the ground as the camera moves.

Overlay style
-------------
Per the Week 3 plan and the user-confirmed choice:
- Past observation (last 10 5-Hz steps of the target): bold blue polyline.
- K=5 predicted samples: semi-transparent red polylines via alpha-blending.
- Mean over K: bold red polyline drawn on top of the K samples.
- Mode marker: small filled circles at the endpoint of each sample.

Usage
-----
Run from the repo root (this file adds its own dir to sys.path, so the
sibling-module imports resolve regardless of cwd). The yolo-deepsort path
needs the detector deps, so use:
    PY=/home/vdelis/anaconda3/envs/new_xupei_env/bin/python
    DRIVER=uav_guidance/server_code_only/main_program/replay_video.py

(1) END-TO-END, zero annotations — YOLO + DeepSORT on raw VisDrone frames.
    Moving camera -> --ego-motion true; scale auto-calibrated from cars.
    Frames may sit directly in the folder OR in an `img/` subfolder.

    $PY $DRIVER \
        --input-images VisDrone2019-MOT-train/sequences/uav0000248_00001_v \
        --anno-format  yolo-deepsort \
        --yolo-weights yolov8x_visdrone.pt \
        --map-model    none \
        --ckpt         tmp/nomap-s-attn/ckpt-best \
        --config       configs/visdrone_eval_5s.py \
        --ego-motion   true \
        --output       tmp/visdrone_e2e/e2e_uav0000248_00001_v.mp4

    # Batch the whole eval set (resolves frames per split automatically):
    #   bash run_visdrone_e2e.sh data/visdrone_levelx/val
    #   bash run_visdrone_e2e.sh data/visdrone_levelx/train 'uav0000248*'

(2) GT-DRIVEN VisDrone replay — boxes from annotations.txt (no detector).
    --scale-mpp is the manual px->m calibration; --map-model none.

    $PY $DRIVER \
        --input-images data/visdrone/uav0000305_00000_v/img \
        --anno-format  visdrone-mot \
        --gt-json      data/visdrone/uav0000305_00000_v/annotations.txt \
        --scale-mpp    0.125 \
        --map-model    none \
        --ckpt         tmp/nomap-s-attn/ckpt-best \
        --config       configs/visdrone_eval_5s.py \
        --ego-motion   true \
        --output       tmp/replay_visdrone_gt.mp4

(3) SYNTHETIC inD clip — orthomap M-ATTN checkpoint, stationary camera.
    --map-pkl triggers PixelMetricShim; --map-model resnet18.

    $PY $DRIVER \
        --input       tmp/synth_uav_clip.mp4 \
        --anno-format synth-json \
        --gt-json     tmp/synth_uav_clip_gt.json \
        --map-pkl     data/levelx/map/inD_01.pkl \
        --map-model   resnet18 \
        --ckpt        tmp/levelx_full_map_v1/ckpt-best \
        --config      configs/levelx_train.py \
        --ego-motion  false \
        --output      tmp/replay_synth_stationary.mp4

Choosing the target vehicle
---------------------------
The auto-picker selects the most-moving agent. Force a specific one with
`--target-aid <id>`. For visdrone-mot, list GT ids with:
    cut -d, -f2 data/visdrone/<seq>/annotations.txt | sort -n | uniq -c
For yolo-deepsort the ids are DeepSORT's own (printed in the run log and the
on-frame HUD), NOT the GT ids.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Local sibling modules (we are running this file directly; the dir is on sys.path).
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from contextvae_inference import ContextVAEInferencer, OB_HORIZON  # noqa: E402
from ego_motion_shim import MovingCameraShim                         # noqa: E402
from pixel_metric_shim import PixelMetricShim, ScaleOnlyShim          # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])

    # ---- frame source: one of --input (mp4) OR --input-images (folder) ----
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to input video (mp4).")
    src.add_argument(
        "--input-images",
        help=(
            "Path to a folder of frame images (jpg/png, sorted by filename). "
            "Use this for VisDrone-MOT sequences whose `img/` subfolder ships "
            "the frames as individual JPEGs rather than an MP4."
        ),
    )

    # ---- annotation source ------------------------------------------------
    p.add_argument(
        "--anno-format", choices=("synth-json", "visdrone-mot", "yolo-deepsort"),
        default="synth-json",
        help=(
            "Where agent bboxes come from. 'synth-json' is the synthetic-clip "
            "JSON written by `synth_clip_gen.py`. 'visdrone-mot' is the "
            "comma-separated VisDroneVDT-2018 MOT annotations.txt. "
            "'yolo-deepsort' runs live YOLOv8 + DeepSORT on the frames (no "
            "annotation file needed — the 'zero annotations' end-to-end path)."
        ),
    )
    p.add_argument(
        "--gt-json", default=None,
        help=(
            "Path to the per-frame agent bbox source. For synth-json this is "
            "the JSON; for visdrone-mot this is the annotations.txt file. "
            "Not used (and rejected) for yolo-deepsort."
        ),
    )

    # ---- yolo-deepsort detector args (only used for --anno-format yolo-deepsort) ---
    p.add_argument(
        "--yolo-weights", default="yolov8x_visdrone.pt",
        help=(
            "Path to the YOLOv8 weights for the yolo-deepsort path. Default "
            "resolves relative to the working directory."
        ),
    )
    p.add_argument(
        "--yolo-conf", type=float, default=0.7,
        help="YOLO confidence threshold for the yolo-deepsort path (default 0.7).",
    )
    p.add_argument(
        "--keep-classes", default=None,
        help=(
            "Comma-separated vehicle class NAMES to keep as trajectory targets "
            "(matched case-insensitively against the model's own names map). "
            "Default: car,van,truck,bus."
        ),
    )

    # ---- shim selection: one of --map-pkl OR --scale-mpp ------------------
    # Not `required` at the group level: the yolo-deepsort path auto-calibrates
    # scale from car detections, so it needs NEITHER. The per-format validation
    # below enforces "exactly one" for synth-json / visdrone-mot.
    pix = p.add_mutually_exclusive_group(required=False)
    pix.add_argument(
        "--map-pkl",
        help=(
            "Path to the orthomap pickle (semantic_map, H) for this recording. "
            "Use for inD/uniD/rounD and any other dataset that ships a "
            "registered orthomap. Triggers PixelMetricShim."
        ),
    )
    pix.add_argument(
        "--scale-mpp", type=float,
        help=(
            "Scalar meters-per-pixel calibration for datasets WITHOUT a "
            "registered orthomap (e.g. VisDroneVDT-2018). Triggers "
            "ScaleOnlyShim. Calibrate from a known in-scene object: e.g. "
            "for a sedan whose bbox is 40 px wide at this altitude, "
            "--scale-mpp 0.1125 (= 4.5 m / 40 px)."
        ),
    )
    p.add_argument(
        "--origin-uv", nargs=2, type=float, metavar=("U", "V"),
        default=None,
        help=(
            "Origin pixel (u, v) for ScaleOnlyShim. Pixel at this location "
            "maps to world (0, 0). Default: frame center (read from the "
            "first frame after open). Ignored when --map-pkl is used."
        ),
    )

    p.add_argument(
        "--ckpt", required=True,
        help="Path to ContextVAE checkpoint (ckpt-best).",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to the training config (configs/levelx_train.py).",
    )
    p.add_argument(
        "--ego-motion", choices=("true", "false"), default="false",
        help=(
            "When 'true', wrap the static shim in MovingCameraShim and run "
            "ORB+RANSAC per frame to compensate for camera motion."
        ),
    )
    p.add_argument("--output", required=True, help="Path to output annotated mp4.")
    p.add_argument(
        "--fps-step", type=int, default=None,
        help=(
            "Number of video frames per 5-Hz history step. Defaults to the "
            "GT JSON's `frame_repeat` field for synth-json; 6 (== 30/5) for "
            "visdrone-mot. Override here if your source has a different fps."
        ),
    )
    p.add_argument(
        "--target-aid", type=int, default=None,
        help=(
            "Optional. Force a specific agent-id as the prediction target. "
            "Defaults to the agent with the longest history present in the "
            "clip."
        ),
    )
    p.add_argument(
        "--map-model", choices=("resnet18", "none"), default="resnet18",
        help=(
            "Override for ContextVAE's map_model arg. 'resnet18' for the "
            "M-ATTN checkpoint; 'none' for the no-map S-ATTN checkpoint. "
            "Must be 'none' when --scale-mpp is set (no orthomap available)."
        ),
    )
    args = p.parse_args()

    # --- per-format bbox-source validation --------------------------------
    if args.anno_format == "yolo-deepsort":
        # Live detector: no annotation file, and an orthomap makes no sense
        # (arbitrary footage, scale auto-calibrated from car detections).
        if args.gt_json is not None:
            p.error("--gt-json is not used with --anno-format yolo-deepsort.")
        if args.map_pkl is not None:
            p.error(
                "--map-pkl is not supported with yolo-deepsort (no registered "
                "orthomap for live footage). Omit it; scale is auto-calibrated, "
                "or pass --scale-mpp to override. Use --map-model none."
            )
        if args.map_model != "none":
            p.error(
                "yolo-deepsort has no orthomap; pass --map-model none "
                "(use the no-map S-ATTN checkpoint)."
            )
    else:
        # File-based sources REQUIRE the annotation file and exactly one shim.
        if args.gt_json is None:
            p.error(f"--gt-json is required for --anno-format {args.anno_format}.")
        if args.map_pkl is None and args.scale_mpp is None:
            p.error(
                f"--anno-format {args.anno_format} needs a shim: pass one of "
                "--map-pkl (orthomap) or --scale-mpp (scalar calibration)."
            )

    # Cross-arg consistency: --scale-mpp implies no map, which means we MUST
    # use the no-map ckpt; --map-pkl REQUIRES the M-ATTN ckpt (or the
    # checkpoint's map_model has to match). Enforce the combo here so a
    # checkpoint/shim mismatch fails at arg-parse rather than at state_dict
    # load time with a confusing shape error.
    if args.scale_mpp is not None and args.map_model != "none":
        p.error("--scale-mpp implies no orthomap; pass --map-model none too.")
    if args.map_pkl is not None and args.map_model == "none":
        p.error(
            "--map-model none with --map-pkl is a mismatched combo: a "
            "PixelMetricShim is constructed for nothing. Either drop "
            "--map-pkl in favor of --scale-mpp, or use --map-model resnet18."
        )
    return args


# ---------------------------------------------------------------------------
# Bbox-source loader (GT JSON)
# ---------------------------------------------------------------------------

def load_gt_bboxes(json_path: str) -> Tuple[
    Dict[int, List[Dict]], int, Tuple[int, int], int, int
]:
    """
    Load per-frame agent bboxes from the synthetic-clip GT JSON.

    Returns
    -------
    by_frame : Dict[int, List[Dict]]
        Maps video-frame-index -> list of agent dicts:
        {"aid": int, "ltwh_live": [l, t, w, h], "center_live": [u, v],
         "world_gt": [x, y] (optional, used only for diagnostic logging)}.
    n_frames : int
    k0_shape_hw : (H, W)
        Reference frame size — used only for diagnostic logging.
    frame_repeat : int
        Inverse of the underlying-5Hz/video-fps ratio.
    fps : int
        Source video fps (informational).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    by_frame: Dict[int, List[Dict]] = {}
    for fr in data["frames"]:
        frame_idx = int(fr["frame"])
        agents = []
        for a in fr["agents"]:
            agents.append({
                "aid": int(a["aid"]),
                "ltwh_live": list(a["ltwh_live"]),
                "center_live": list(a["center_live"]),
                "world_gt": list(a.get("world", [])),
            })
        by_frame[frame_idx] = agents

    return (
        by_frame,
        int(data["n_frames"]),
        tuple(data["k0_shape_hw"]),
        int(data.get("frame_repeat", 5)),
        int(data.get("fps", 25)),
    )


# ---------------------------------------------------------------------------
# Bbox-source loader (VisDrone-MOT annotations.txt)
# ---------------------------------------------------------------------------

# VisDrone object_category codes that count as "vehicles" for our purposes.
# See the VisDrone-MOT format spec:
#   0  ignored regions      6  truck
#   1  pedestrian           7  tricycle
#   2  people               8  awning-tricycle
#   3  bicycle              9  bus
#   4  car                  10 motor
#   5  van                  11 others
_VISDRONE_VEHICLE_CLASSES = frozenset({4, 5, 6, 9})


def load_visdrone_bboxes(
    anno_path: str,
    img_folder: str,
) -> Tuple[Dict[int, List[Dict]], int, Tuple[int, int], int, int, Dict[int, List[Tuple[float, float, float, float]]]]:
    """
    Parse VisDrone-MOT `annotations.txt` and pair with `img/` to mimic the
    synth-json schema `load_gt_bboxes` returns.

    VisDrone-MOT row format (CSV, no header):
        frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
        score, object_category, truncation, occlusion

    Frame indices in VisDrone are 1-indexed; we convert to 0-indexed to
    match the synth-json (and `cv2.VideoCapture`) convention used elsewhere
    in this driver.

    Returns
    -------
    by_frame : Dict[int, List[Dict]]
        VEHICLE-class agents only. Same shape as `load_gt_bboxes`. Each
        agent dict carries {aid, ltwh_live, center_live, world_gt=None}.
        `world_gt=None` signals the replay loop to SKIP the post-hoc
        world-error diagnostic, since VisDrone has no ground-truth world
        frame.
    n_frames : int
        Number of JPG/PNG frames discovered in `img_folder`.
    k0_shape_hw : (H, W)
        Read from the first frame on disk (informational only).
    frame_repeat : int
        Defaults to 6 (VisDrone is typically 30 fps; ContextVAE wants 5 Hz).
        Caller can override via `--fps-step`.
    fps : int
        Defaults to 30. Informational only.
    mask_bboxes_by_frame : Dict[int, List[(l, t, w, h)]]
        ALL annotated bboxes per frame (vehicles + pedestrians + motors +
        ignored regions etc.), used to mask the ORB feature detector in
        `MovingCameraShim.update_frame`. Critical: if we mask only the
        vehicle bboxes, ORB picks up features on pedestrians and motors,
        those features don't match across frames, and the RANSAC inlier
        ratio collapses -> 86 %+ of frames go FROZEN. By masking ALL
        annotated moving objects, ORB is constrained to the static
        background and the per-frame homography fit converges.
    """
    import glob

    # 1) Frame manifest from the image folder.
    img_exts = ("*.jpg", "*.jpeg", "*.png")
    paths: List[str] = []
    for ext in img_exts:
        paths.extend(glob.glob(os.path.join(img_folder, ext)))
    if not paths:
        raise FileNotFoundError(
            f"No jpg/jpeg/png images found in {img_folder!r}. "
            "VisDrone-MOT sequences ship frames under an `img/` subfolder."
        )
    paths.sort()
    n_frames = len(paths)
    # Read the first frame to learn (H, W). cv2 returns (H, W, C).
    first = cv2.imread(paths[0])
    if first is None:
        raise RuntimeError(f"cv2 failed to decode {paths[0]}")
    k0_shape_hw = (int(first.shape[0]), int(first.shape[1]))

    # 2) Parse the annotations.
    by_frame: Dict[int, List[Dict]] = {}
    mask_bboxes_by_frame: Dict[int, List[Tuple[float, float, float, float]]] = {}
    n_dropped_class = 0
    n_kept = 0
    with open(anno_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            # Some VisDrone splits ship rows with trailing whitespace or
            # empty trailing fields; be lenient on the count beyond 8 but
            # require at least the first 8 columns.
            if len(parts) < 8:
                continue
            frame_1idx = int(parts[0])
            aid = int(parts[1])
            l, t, w, h = (float(parts[i]) for i in range(2, 6))
            klass = int(parts[7])
            frame_0idx = frame_1idx - 1  # VisDrone -> driver convention
            # EVERY annotated bbox goes into the ORB mask, regardless of
            # class — see docstring for the rationale.
            mask_bboxes_by_frame.setdefault(frame_0idx, []).append((l, t, w, h))
            if klass not in _VISDRONE_VEHICLE_CLASSES:
                n_dropped_class += 1
                continue
            cu = l + w / 2.0
            cv_ = t + h / 2.0  # avoid shadowing `cv` (the cv2 alias)
            by_frame.setdefault(frame_0idx, []).append({
                "aid": aid,
                "ltwh_live": [l, t, w, h],
                "center_live": [cu, cv_],
                # VisDrone has no GT world frame; the diagnostic skips
                # world-err accumulation when this field is None.
                "world_gt": None,
            })
            n_kept += 1

    print(
        f"[replay] visdrone-mot: parsed {n_kept} vehicle bboxes "
        f"({n_dropped_class} non-vehicle rows kept for ORB mask only) across "
        f"{len(by_frame)} of {n_frames} frames"
    )
    return by_frame, n_frames, k0_shape_hw, 6, 30, mask_bboxes_by_frame


# ---------------------------------------------------------------------------
# Image-folder frame source
# ---------------------------------------------------------------------------

class ImageFolderCapture:
    """
    Mimics the subset of `cv2.VideoCapture` we actually use, but iterates
    over a sorted list of image files instead of an MP4. Used by the
    VisDrone path (`--input-images`).

    Methods provided:
      - read() -> (ok, frame_bgr)
      - get(cv2.CAP_PROP_FPS)        -> assumed_fps (default 30)
      - get(cv2.CAP_PROP_FRAME_WIDTH)  -> width from first frame
      - get(cv2.CAP_PROP_FRAME_HEIGHT) -> height from first frame
      - release()
      - isOpened()
    """

    def __init__(self, folder: str, assumed_fps: float = 30.0):
        import glob

        def _glob_frames(d: str) -> List[str]:
            out: List[str] = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                out.extend(glob.glob(os.path.join(d, ext)))
            return out

        # Accept BOTH on-disk layouts transparently:
        #   <folder>/*.jpg            (raw VisDrone splits: sequences/<seq>/*.jpg)
        #   <folder>/img/*.jpg        (reorganized: data/visdrone/<seq>/img/*.jpg)
        # If the folder itself has no frames but an `img/` subdir does, descend
        # into it. This is what lets a visdrone_levelx-driven run point at the
        # bare sequence dir regardless of which layout that split uses.
        paths = _glob_frames(folder)
        if not paths and os.path.isdir(os.path.join(folder, "img")):
            paths = _glob_frames(os.path.join(folder, "img"))
        if not paths:
            raise FileNotFoundError(
                f"No frames in {folder!r} (looked in the folder and in img/)."
            )
        paths.sort()
        self._paths = paths
        self._idx = 0
        # Probe the first frame for shape WITHOUT consuming it (we re-read
        # in read() below). Acceptable to pay the decode cost once.
        first = cv2.imread(paths[0])
        if first is None:
            raise RuntimeError(f"cv2 failed to decode {paths[0]}")
        self._h, self._w = first.shape[:2]
        self._fps = float(assumed_fps)

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._idx >= len(self._paths):
            return False, None
        frame = cv2.imread(self._paths[self._idx])
        self._idx += 1
        if frame is None:
            return False, None
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        return 0.0

    def release(self):
        # No file handles held open between read() calls.
        pass


# ---------------------------------------------------------------------------
# Bbox-source loader (live YOLOv8 + DeepSORT — the "zero annotations" path)
# ---------------------------------------------------------------------------

# Scale-calibration constants, mirrored from preprocessing/process_visdrone.py
# so the end-to-end path produces the same metric scale as the GT-based eval.
_ASSUMED_CAR_LEN_M = 4.5      # sedan length used to turn px -> metres
_DEFAULT_MPP = 0.045         # hard fallback m/px if no car ever detected
# Default vehicle class NAMES (lowercase). Resolved against the model's own
# `names` map at load time — NOT hardcoded ids — because a "visdrone" YOLO
# checkpoint may actually carry COCO class indices (car=2, bus=5, truck=7,
# and no "van"), so id {4,5,6,9} would select the wrong classes entirely.
_DEFAULT_KEEP_CLASS_NAMES = frozenset({"car", "van", "truck", "bus"})


def load_yolo_deepsort_bboxes(
    input_path: str,
    is_image_folder: bool,
    weights_path: str,
    conf: float,
    keep_class_names: frozenset,
    assumed_fps: float = 30.0,
    scale_override: Optional[float] = None,
) -> Tuple[
    Dict[int, List[Dict]], int, Tuple[int, int], int, int,
    Dict[int, List[Tuple[float, float, float, float]]], float,
]:
    """
    Run YOLOv8 + DeepSORT over a frame source as a PRE-PASS and emit the same
    `by_frame` schema the GT loaders return — but with boxes produced live by
    the detector instead of read from an annotation file. This is the
    "zero annotations" end-to-end path (YOLO detection -> ContextVAE).

    Why a pre-pass (not interleaved with the render loop)?
    -----------------------------------------------------
    `pick_target_aid` (called in main() before the render loop starts) scans
    the FULL `by_frame` to choose the target. A detector running inside the
    render loop couldn't populate `by_frame` in time. DeepSORT also needs
    frames in temporal order, which a single forward pass gives us anyway.

    Detection helper
    ----------------
    We deliberately do NOT reuse `scene_recog_socket_yolov8.get_boxes_feats_yolov8`:
    that calls `model(img, conf=.., return_class_logits=True)` — a Ntousis
    ultralytics patch whose `f_lst` output exists only for the re-ID
    similarity map, which replay does not need. We call the model plainly and
    build the `(ltwh, conf, cls)` tuples DeepSORT consumes.

    Class resolution
    ----------------
    Vehicle classes are resolved by NAME from `model.names` (see
    `_DEFAULT_KEEP_CLASS_NAMES`). Cars (specifically) drive the auto-scale.

    Returns
    -------
    Same 6 fields as `load_visdrone_bboxes`, PLUS a 7th:
    by_frame, n_frames, k0_shape_hw, frame_repeat, fps, mask_bboxes_by_frame,
    m_per_px
        `mask_bboxes_by_frame` holds EVERY raw detection (all classes) so the
        ORB feature mask in ego-motion mode excludes all moving objects — the
        same all-bboxes policy `load_visdrone_bboxes` uses to avoid the ORB
        fit freezing. `m_per_px` is the auto-calibrated scale (or the override).
    """
    # Lazy imports: only the yolo-deepsort path needs these heavy deps, so the
    # synth-json / visdrone-mot paths keep working without them installed.
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort

    model = YOLO(weights_path)
    names = {int(cid): str(nm) for cid, nm in model.names.items()}
    vehicle_ids = {cid for cid, nm in names.items() if nm.lower() in keep_class_names}
    car_ids = {cid for cid, nm in names.items() if nm.lower() == "car"}
    if not vehicle_ids:
        raise SystemExit(
            f"[replay] yolo-deepsort: none of {sorted(keep_class_names)} match "
            f"the model's class names {names}. Pass --keep-classes with names "
            "that exist in this checkpoint."
        )
    print(
        "[replay] yolo-deepsort: weights={} resolved vehicle classes {}".format(
            os.path.basename(weights_path),
            {cid: names[cid] for cid in sorted(vehicle_ids)},
        )
    )

    # Frame source (same two backends as the render loop).
    if is_image_folder:
        cap = ImageFolderCapture(input_path, assumed_fps=assumed_fps)
    else:
        cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"[replay] yolo-deepsort: could not open {input_path!r}")

    tracker = DeepSort(max_age=3, n_init=2)

    by_frame: Dict[int, List[Dict]] = {}
    mask_bboxes_by_frame: Dict[int, List[Tuple[float, float, float, float]]] = {}
    car_long_axes: List[float] = []
    h_img = w_img = 0
    n_det_total = 0
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if h_img == 0:
            h_img, w_img = frame.shape[:2]

        # --- detect (plain call; no return_class_logits / no f_lst) ---------
        res = model(frame, conf=conf, verbose=False)[0]
        results: List[Tuple[List[float], float, int]] = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                ltwh = [x1, y1, x2 - x1, y2 - y1]
                results.append((ltwh, float(confs[i]), int(clss[i])))
        n_det_total += len(results)

        # EVERY raw detection feeds the ORB mask (ego-motion robustness).
        mask_bboxes_by_frame[frame_idx] = [tuple(d[0]) for d in results]

        # Vehicle subset goes to the tracker; cars also drive the scale calib.
        veh = [d for d in results if d[2] in vehicle_ids]
        for ltwh, _cf, cls in veh:
            if cls in car_ids:
                car_long_axes.append(max(ltwh[2], ltwh[3]))

        # DeepSORT: pass the FRAME via the `frame=` kwarg so the mobilenet
        # embedder computes appearance features. (The Ntousis server passes it
        # positionally, where it lands in `embeds` — a latent bug we avoid.)
        tracks = tracker.update_tracks(veh, frame=frame)
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            l, t, w, h = (float(v) for v in tr.to_ltwh())
            by_frame.setdefault(frame_idx, []).append({
                "aid": int(tr.track_id),
                "ltwh_live": [l, t, w, h],
                "center_live": [l + w / 2.0, t + h / 2.0],
                # No GT world frame for live detections — disables the
                # post-hoc world-error diagnostic in the render loop.
                "world_gt": None,
            })
    cap.release()
    n_frames = frame_idx + 1

    # --- auto-scale: m/px = assumed car length / median car-bbox long axis ---
    if scale_override is not None:
        m_per_px = float(scale_override)
        scale_src = "override"
    elif car_long_axes:
        m_per_px = _ASSUMED_CAR_LEN_M / float(np.median(car_long_axes))
        scale_src = f"car-median(n={len(car_long_axes)})"
    else:
        m_per_px = _DEFAULT_MPP
        scale_src = "default(no cars detected)"

    n_tracks = len({a["aid"] for agents in by_frame.values() for a in agents})
    print(
        "[replay] yolo-deepsort: {} frames  {} detections  {} confirmed tracks  "
        "m_per_px={:.5f} ({})".format(
            n_frames, n_det_total, n_tracks, m_per_px, scale_src,
        )
    )
    return (
        by_frame, n_frames, (h_img, w_img), 6, int(round(assumed_fps)),
        mask_bboxes_by_frame, m_per_px,
    )


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def pick_target_aid(by_frame: Dict[int, List[Dict]], forced: Optional[int]) -> int:
    """
    Pick the agent-id to use as the ContextVAE target.

    If `forced` is given, returns it (raises if not present at all).

    Otherwise scores each agent by `appearances * total_pixel_motion` and
    picks the winner. The motion term is important: an auto-picker that
    only counts appearances will happily pick a car that's STOPPED at a
    red light for the entire observation window, in which case the
    driver's atan2-finite-difference heading estimate collapses to
    pure noise and the model predicts in an arbitrary direction. Cars
    that are actually driving give a clean heading and visually
    meaningful predictions.

    Pixel-space motion is noisy on moving-camera footage (mixes agent
    motion with camera motion), but the noise is symmetric: agents that
    are TRULY stationary score near zero regardless of camera pan,
    while agents that are moving score high because their pixel
    displacement includes their world motion. Good enough for an
    auto-pick; the user can always override via `--target-aid`.
    """
    if forced is not None:
        # Pre-collect the universe of aids so we can give a useful error
        # when forced doesn't exist.
        seen = {a["aid"] for agents in by_frame.values() for a in agents}
        if forced not in seen:
            raise SystemExit(
                f"--target-aid {forced} not found in GT JSON; "
                f"available aids: {sorted(seen)}"
            )
        return forced

    # Build per-aid time series sorted by frame_idx, then compute
    # cumulative pixel displacement.
    series: Dict[int, List[Tuple[int, float, float]]] = collections.defaultdict(list)
    for fr_idx, agents in by_frame.items():
        for a in agents:
            cu, cv_ = a["center_live"]
            series[a["aid"]].append((int(fr_idx), float(cu), float(cv_)))

    scores: List[Tuple[float, int]] = []
    for aid, pts in series.items():
        pts.sort(key=lambda p: p[0])
        if len(pts) < 2:
            continue
        total_disp = 0.0
        for i in range(1, len(pts)):
            du = pts[i][1] - pts[i - 1][1]
            dv = pts[i][2] - pts[i - 1][2]
            total_disp += math.hypot(du, dv)
        # Composite score. The product matters more than the formula —
        # both terms must be non-trivial for the agent to be a useful
        # target. Adding 1 to total_disp avoids 0-score elimination of
        # stationary-but-long-lived agents (still better to pick a
        # stopped car that exists everywhere than nothing).
        scores.append((len(pts) * (total_disp + 1.0), aid))

    if not scores:
        raise SystemExit(
            "pick_target_aid: no agent has 2+ appearances; nothing to "
            "predict on. Check the annotation file."
        )
    # Sort by (-score, aid): highest score wins, ties broken low.
    scores.sort(key=lambda sa: (-sa[0], sa[1]))
    return scores[0][1]


# ---------------------------------------------------------------------------
# History maintenance
# ---------------------------------------------------------------------------

class AgentHistory:
    """
    Per-agent ring buffer of (x_m, y_m) world positions, length OB_HORIZON.

    We store world coords directly (the shim does the live->world conversion
    before append). This means an ego-motion shim's per-frame H is applied
    EAGERLY at append time, and the cached history doesn't have to be
    re-projected when the camera moves further.
    """

    def __init__(self) -> None:
        self._buf: Deque[Tuple[float, float]] = collections.deque(maxlen=OB_HORIZON)

    def push(self, x_m: float, y_m: float) -> None:
        self._buf.append((float(x_m), float(y_m)))

    def ready(self) -> bool:
        return len(self._buf) >= OB_HORIZON

    def as_array(self) -> np.ndarray:
        return np.asarray(self._buf, dtype=np.float64)

    def heading_estimate(self) -> Optional[float]:
        """First-obs-frame heading from finite difference (atan2).

        Returns None if too few samples to estimate; otherwise radians
        in world frame. We use this in lieu of a true heading source,
        accepting the documented degradation on stationary agents. For
        the synthetic-clip target (a moving vehicle) this is fine.
        """
        if len(self._buf) < 2:
            return None
        x0, y0 = self._buf[0]
        x1, y1 = self._buf[1]
        return math.atan2(y1 - y0, x1 - x0)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

# BGR colors (cv2 convention).
_PAST_COLOR    = (255, 80, 80)    # blue-ish (BGR)
_PRED_COLOR    = (0, 0, 255)      # red
_MEAN_COLOR    = (0, 0, 255)      # red, drawn bolder than samples
_PAST_THICK    = 2
_SAMPLE_THICK  = 1
_MEAN_THICK    = 2
_SAMPLE_ALPHA  = 0.35
_ENDPOINT_RADIUS = 4


def draw_overlay(
    frame_bgr: np.ndarray,
    past_uv: Optional[np.ndarray],
    samples_uv: Optional[np.ndarray],
    mean_uv: Optional[np.ndarray],
) -> np.ndarray:
    """
    Render past+K=5+mean overlay onto a copy of the frame.

    Parameters
    ----------
    frame_bgr   : (H, W, 3) uint8 BGR frame.
    past_uv     : (OB_HORIZON, 2) live-pixel coords of the target's past, or None.
    samples_uv  : (K, T_pred, 2) live-pixel coords of the K predicted samples, or None.
    mean_uv     : (T_pred, 2) live-pixel coords of the mean prediction, or None.

    Returns
    -------
    annotated : (H, W, 3) uint8 BGR — a new image; input not mutated.
    """
    out = frame_bgr.copy()
    H_img, W_img = out.shape[:2]

    # 1) Past polyline (no alpha).
    if past_uv is not None and len(past_uv) >= 2:
        pts = past_uv.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=_PAST_COLOR,
                      thickness=_PAST_THICK, lineType=cv2.LINE_AA)

    # 2) K=5 predicted samples — alpha-blended.
    # Draw on an overlay copy, then addWeighted into `out`. This keeps the
    # alpha cheap (one addWeighted) and looks clean regardless of how many
    # samples overlap.
    if samples_uv is not None and samples_uv.shape[1] >= 2:
        overlay = out.copy()
        for k in range(samples_uv.shape[0]):
            pts = samples_uv[k].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=False, color=_PRED_COLOR,
                          thickness=_SAMPLE_THICK, lineType=cv2.LINE_AA)
            # Endpoint dot per sample.
            ex, ey = int(samples_uv[k, -1, 0]), int(samples_uv[k, -1, 1])
            if 0 <= ex < W_img and 0 <= ey < H_img:
                cv2.circle(overlay, (ex, ey), _ENDPOINT_RADIUS,
                           _PRED_COLOR, thickness=-1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, _SAMPLE_ALPHA, out, 1.0 - _SAMPLE_ALPHA, 0.0,
                        dst=out)

    # 3) Mean polyline — opaque, drawn ON TOP of the alpha-blended samples
    # so it remains the most visible track.
    if mean_uv is not None and len(mean_uv) >= 2:
        pts = mean_uv.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=_MEAN_COLOR,
                      thickness=_MEAN_THICK, lineType=cv2.LINE_AA)

    return out


def draw_hud(
    frame_bgr: np.ndarray,
    frame_idx: int,
    n_frames: int,
    target_aid: int,
    mode_label: str,
    last_status: str,
    last_inf_ms: Optional[float],
    failure_banner: Optional[str] = None,
) -> None:
    """
    Draw a one-line status HUD at the top-left, mutating `frame_bgr`.

    When `failure_banner` is provided, also draws a large red banner across
    the top of the frame to make the "this output is intentionally wrong"
    nature of the stationary baseline self-explanatory to a Ch6 reader who
    can't see the diagnostic numbers in the terminal.
    """
    # --- Top-of-frame banner (stationary mode only) ---------------------
    # We render this BEFORE the status line so the small status text sits
    # below it. The banner is intentionally heavy-weight: large red text
    # with a black outline so it survives JPEG/video compression.
    if failure_banner:
        H_img, W_img = frame_bgr.shape[:2]
        # Center horizontally; sit just below the top edge.
        (tw, th), _ = cv2.getTextSize(
            failure_banner, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2,
        )
        bx = max(10, (W_img - tw) // 2)
        by = th + 14
        cv2.putText(
            frame_bgr, failure_banner, (bx, by),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA,  # outline
        )
        cv2.putText(
            frame_bgr, failure_banner, (bx, by),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA,  # red
        )

    label = f"f={frame_idx:3d}/{n_frames-1}  tgt={target_aid}  {mode_label}"
    if last_status:
        label += f"  shim={last_status}"
    if last_inf_ms is not None:
        label += f"  inf={last_inf_ms:.1f}ms"
    # Push the status line further down when the banner is present so the
    # two don't overlap.
    y_status = 60 if failure_banner else 22
    cv2.putText(
        frame_bgr, label, (10, y_status),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,  # outline
    )
    cv2.putText(
        frame_bgr, label, (10, y_status),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    use_ego = args.ego_motion == "true"
    mode_label = "ego-motion" if use_ego else "stationary"
    print(f"[replay] mode = {mode_label}")
    print(f"[replay] input        = {args.input or args.input_images}")
    print(f"[replay] anno-format  = {args.anno_format}")
    print(f"[replay] gt           = {args.gt_json}")
    if args.map_pkl:
        shim_src = f"map-pkl {args.map_pkl}"
    elif args.scale_mpp is not None:
        shim_src = f"scale-mpp {args.scale_mpp}"
    elif args.anno_format == "yolo-deepsort":
        shim_src = "scale-mpp auto (yolo car-detection calibration)"
    else:
        shim_src = "(none)"
    print(f"[replay] shim-source  = {shim_src}")
    print(f"[replay] ckpt         = {args.ckpt}")

    # --- Bbox stream ----------------------------------------------------
    # Load first so we know `n_frames` and the source `(H, W)` for the
    # scale-only origin default.
    # `mask_bboxes_by_frame` is the union of ALL annotated bboxes per
    # frame (vehicles + pedestrians + motors + ignored regions). It's
    # used ONLY for masking the ORB feature detector in ego-motion mode;
    # the trackable agents fed to ContextVAE come from `by_frame` which
    # is vehicle-only. For the synth path we fall back to the vehicle
    # bboxes (no separate non-vehicle annotations exist in that schema).
    mask_bboxes_by_frame: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None
    # Scale auto-calibrated by the yolo-deepsort detector pre-pass; None for the
    # file-based sources (which carry an explicit --scale-mpp or --map-pkl).
    auto_scale_mpp: Optional[float] = None
    if args.anno_format == "synth-json":
        by_frame, n_frames, k0_shape_hw, frame_repeat, fps_src = load_gt_bboxes(
            args.gt_json
        )
    elif args.anno_format == "visdrone-mot":
        if args.input_images is None:
            print(
                "[replay] ERROR: --anno-format visdrone-mot requires "
                "--input-images <folder>",
                file=sys.stderr,
            )
            return 2
        (
            by_frame, n_frames, k0_shape_hw, frame_repeat, fps_src,
            mask_bboxes_by_frame,
        ) = load_visdrone_bboxes(args.gt_json, args.input_images)
    else:  # yolo-deepsort (live detection; no annotation file)
        # Accepts either --input (mp4) or --input-images (folder).
        is_folder = args.input_images is not None
        src_path = args.input_images if is_folder else args.input
        keep_names = (
            frozenset(s.strip().lower() for s in args.keep_classes.split(",") if s.strip())
            if args.keep_classes else _DEFAULT_KEEP_CLASS_NAMES
        )
        (
            by_frame, n_frames, k0_shape_hw, frame_repeat, fps_src,
            mask_bboxes_by_frame, auto_scale_mpp,
        ) = load_yolo_deepsort_bboxes(
            input_path=src_path,
            is_image_folder=is_folder,
            weights_path=args.yolo_weights,
            conf=args.yolo_conf,
            keep_class_names=keep_names,
            scale_override=args.scale_mpp,
        )
    fps_step = args.fps_step if args.fps_step is not None else frame_repeat
    target_aid = pick_target_aid(by_frame, args.target_aid)
    print(
        f"[replay] n_frames={n_frames} fps={fps_src} frame_repeat={frame_repeat} "
        f"fps_step={fps_step}  target_aid={target_aid}"
    )

    # --- Inferencer ----------------------------------------------------
    # The argparse cross-arg check already ensured (--scale-mpp <-> --map-model none)
    # and (--map-pkl <-> --map-model resnet18) are consistent.
    map_model_override = None if args.map_model == "none" else args.map_model
    inferencer = ContextVAEInferencer(
        ckpt_path=args.ckpt,
        config_path=args.config,
        map_pickle_path=args.map_pkl,  # None when --scale-mpp is set
        device=None,
        model_overrides={"map_model": map_model_override},
    )
    print(
        f"[replay] loaded ckpt epoch={inferencer._loaded_epoch} "
        f"ADE={inferencer._loaded_ade:.4f} FDE={inferencer._loaded_fde:.4f}"
    )

    # --- Shim ----------------------------------------------------------
    # Build the STATIC base shim (homography or scale-only). When
    # --ego-motion=true, wrap it in MovingCameraShim for per-frame ORB+RANSAC
    # compensation. The wrap is uniform across both base shim types thanks
    # to the four-method API contract enforced earlier.
    if args.map_pkl is not None:
        base_shim = PixelMetricShim(args.map_pkl)
    else:
        # ScaleOnlyShim: default origin to frame center (read from bbox source).
        # Scale priority: explicit --scale-mpp > yolo-deepsort auto-calibration.
        # (For yolo-deepsort the loader already folded --scale-mpp into
        # `auto_scale_mpp` when present, so a single source of truth remains.)
        h_img, w_img = k0_shape_hw
        if args.origin_uv is not None:
            origin_uv = (float(args.origin_uv[0]), float(args.origin_uv[1]))
        else:
            origin_uv = (w_img / 2.0, h_img / 2.0)
        effective_mpp = args.scale_mpp if args.scale_mpp is not None else auto_scale_mpp
        if effective_mpp is None:
            print(
                "[replay] ERROR: no pixel->metric scale available (need "
                "--scale-mpp, --map-pkl, or the yolo-deepsort auto-calibration).",
                file=sys.stderr,
            )
            return 2
        base_shim = ScaleOnlyShim(
            m_per_px=float(effective_mpp), origin_uv=origin_uv,
        )
        print(
            f"[replay] ScaleOnlyShim: m_per_px={base_shim.m_per_px:.6f} "
            f"origin_uv=({base_shim.u0:.1f}, {base_shim.v0:.1f})"
        )

    if use_ego:
        # Compose: ORB+RANSAC for live->K0, base shim for K0->world.
        shim = MovingCameraShim(base_shim=base_shim)
    else:
        shim = base_shim

    # --- Frame source --------------------------------------------------
    # Either an MP4 (cv2.VideoCapture) or a folder of JPEGs/PNGs
    # (ImageFolderCapture). The two implement the same minimal interface.
    if args.input is not None:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = ImageFolderCapture(args.input_images, assumed_fps=float(fps_src))
    if not cap.isOpened():
        src_name = args.input or args.input_images
        print(f"[replay] ERROR: could not open {src_name}", file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps_src)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        print(f"[replay] ERROR: could not open writer for {args.output}",
              file=sys.stderr)
        return 2

    # --- Per-track state ----------------------------------------------
    histories: Dict[int, AgentHistory] = {}
    # Cache for the most recent inference output, drawn over every frame
    # until the next 5-Hz tick fires. Stored in WORLD coords so we can
    # re-project per frame as the camera moves.
    cached_samples_world: Optional[np.ndarray] = None   # (K, T_pred, 2)
    cached_past_world:    Optional[np.ndarray] = None   # (OB_HORIZON, 2)

    # --- Diagnostics --------------------------------------------------
    inf_latencies_ms: List[float] = []
    n_inferences = 0
    last_inf_ms: Optional[float] = None
    last_shim_status: str = ""
    # Per-frame world-position error of the target (for the ego-motion
    # validation criterion in the plan).
    pos_errors_m: List[float] = []

    frame_idx = -1
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx >= n_frames:
            break

        # 1) Update the shim for this frame (ego-motion mode only). We
        # pass the union of agent bboxes so ORB excludes moving objects.
        # On VisDrone we use the full annotation set (vehicles + peds +
        # motors + ignored) — masking only the vehicle subset leaves
        # pedestrians/motors free to inject moving features into the
        # match pool, which empirically pushes 86 % of frames to FROZEN.
        # On synth-json there are no non-vehicle annotations, so the
        # vehicle list IS the full annotation set and we use it directly.
        if use_ego:
            if mask_bboxes_by_frame is not None:
                mask_bboxes = list(mask_bboxes_by_frame.get(frame_idx, []))
            else:
                agents_this_frame = by_frame.get(frame_idx, [])
                mask_bboxes = [a["ltwh_live"] for a in agents_this_frame]
            last_shim_status = shim.update_frame(frame_bgr, mask_bboxes_ltwh=mask_bboxes)

        # 2) Append to per-agent histories at the 5-Hz cadence. We pick
        # the BBOX CENTER as the agent's "position" and convert via the
        # shim. Anchoring on the bbox center (not the corner) matches
        # how the levelXdata preprocessing stored xCenter/yCenter.
        if frame_idx % fps_step == 0:
            agents_this_frame = by_frame.get(frame_idx, [])
            for a in agents_this_frame:
                aid = a["aid"]
                u, v = a["center_live"]
                try:
                    x_m, y_m = shim.pixel_to_metric(float(u), float(v))
                except RuntimeError:
                    # Degenerate homography (ego-motion frozen at startup, etc.).
                    continue
                histories.setdefault(aid, AgentHistory()).push(x_m, y_m)

                # Diagnostic: GT world position is available in the JSON;
                # accumulate position error for the target only.
                if aid == target_aid and a.get("world_gt"):
                    xg, yg = a["world_gt"]
                    pos_errors_m.append(float(math.hypot(x_m - xg, y_m - yg)))

            # 3) Run inference if the target has a full window.
            target_hist = histories.get(target_aid)
            if target_hist is not None and target_hist.ready():
                # Build neighbor dict from every OTHER aid that has 10 frames.
                neighbors = {
                    aid: hist.as_array()
                    for aid, hist in histories.items()
                    if aid != target_aid and hist.ready()
                }
                t0 = time.perf_counter()
                samples_world = inferencer.infer_samples(
                    target_history_metric=target_hist.as_array(),
                    neighbor_histories_metric=neighbors,
                    heading_world=target_hist.heading_estimate(),
                )  # (K, T_pred, 2)
                last_inf_ms = (time.perf_counter() - t0) * 1000.0
                inf_latencies_ms.append(last_inf_ms)
                n_inferences += 1
                cached_samples_world = samples_world
                cached_past_world = target_hist.as_array().copy()

        # 4) Render: convert cached world-frame past + K samples to live
        # pixels using the CURRENT frame's shim, then draw.
        past_uv = None
        samples_uv = None
        mean_uv = None
        if cached_past_world is not None and cached_samples_world is not None:
            try:
                past_uv = shim.metrics_to_pixel(cached_past_world)
                K, T_pred, _ = cached_samples_world.shape
                samples_flat = cached_samples_world.reshape(K * T_pred, 2)
                samples_uv = shim.metrics_to_pixel(samples_flat).reshape(K, T_pred, 2)
                mean_world = cached_samples_world.mean(axis=0)
                mean_uv = shim.metrics_to_pixel(mean_world)
            except RuntimeError:
                # Degenerate H in this frame — skip the overlay this frame.
                past_uv = samples_uv = mean_uv = None

        annotated = draw_overlay(frame_bgr, past_uv, samples_uv, mean_uv)
        # Loud failure-mode banner ONLY in stationary mode — its overlays
        # are expected to drift off-road when the camera moves, and the
        # banner makes that intent legible to a thesis reader who isn't
        # cross-referencing the terminal diagnostics.
        failure_banner = (
            "STATIONARY BASELINE - EXPECTED DRIFT (no ego-motion compensation)"
            if not use_ego else None
        )
        draw_hud(annotated, frame_idx, n_frames, target_aid, mode_label,
                 last_shim_status, last_inf_ms, failure_banner=failure_banner)
        writer.write(annotated)

    cap.release()
    writer.release()

    # --- Summary -------------------------------------------------------
    print(f"[replay] wrote {args.output}")
    print(f"[replay] inferences  : {n_inferences}")
    if inf_latencies_ms:
        arr = np.asarray(inf_latencies_ms)
        print(
            f"[replay] inf latency : "
            f"median={np.median(arr):.2f} ms  "
            f"p95={np.percentile(arr, 95):.2f} ms  "
            f"max={arr.max():.2f} ms"
        )
    if pos_errors_m:
        arr = np.asarray(pos_errors_m)
        print(
            f"[replay] world-err   : "
            f"median={np.median(arr):.4f} m  "
            f"max={arr.max():.4f} m  "
            f"(target_aid={target_aid}, n={len(arr)})"
        )
    if use_ego and isinstance(shim, MovingCameraShim):
        print(
            f"[replay] shim stats  : frames_seen={shim.n_frames_seen} "
            f"reanchors={shim.n_reanchors} frozen={shim.n_frozen}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
