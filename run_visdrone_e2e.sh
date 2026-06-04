#!/usr/bin/env bash
# ============================================================================
# run_visdrone_e2e.sh
# End-to-end (YOLO detection -> DeepSORT tracking -> ContextVAE prediction)
# QUALITATIVE replay over VisDrone sequences. For each sequence it runs the
# full detection-driven pipeline (no ground-truth annotations) and writes an
# annotated MP4 with the tracked target's past + ContextVAE K=5 prediction
# overlays. Pixel->metric scale is auto-calibrated from YOLO car detections.
#
# This is the deployment-faithful counterpart to run_visdrone_eval.sh (which
# scores ContextVAE on GROUND-TRUTH tracks).
#
# TWO input modes — auto-detected from the first argument:
#
#   (A) visdrone_levelx split dir  (contains <seq>.txt files)
#       Enumerates the SAME sequences the eval uses, and resolves each one's
#       raw frames from the matching VisDrone image split:
#           train     -> VisDrone2019-MOT-train/sequences/<seq>/
#           test-dev  -> VisDrone2019-MOT-test-dev/sequences/<seq>/
#           val       -> full-visdrone/sequences/<seq>/
#       (override any of these via the *_IMG_ROOT env vars, or force a single
#        root for all seqs via IMG_ROOT=...).
#         bash run_visdrone_e2e.sh data/visdrone_levelx/val
#         bash run_visdrone_e2e.sh data/visdrone_levelx/train 'uav0000248*'
#
#   (B) image root  (contains <seq>/ subfolders of frames)
#       Frames may sit directly in <seq>/ OR in <seq>/img/ — both work.
#         bash run_visdrone_e2e.sh data/visdrone
#         bash run_visdrone_e2e.sh VisDrone2019-MOT-train/sequences 'uav0000361*'
#
# Useful env overrides:
#   CKPT= CONFIG= YOLO_WEIGHTS= YOLO_CONF= EGO_MOTION= OUTDIR=
#   IMG_ROOT=  TRAIN_IMG_ROOT=  TESTDEV_IMG_ROOT=  VAL_IMG_ROOT=
#   DRY_RUN=1   # just list resolved seq -> frame_dir and exit (no rendering)
# ============================================================================
set -uo pipefail

PROJ=/home/vdelis/ContextVAE
PY=/home/vdelis/anaconda3/envs/new_xupei_env/bin/python
DRIVER=uav_guidance/server_code_only/main_program/replay_video.py

# --- inputs -----------------------------------------------------------------
SRC="${1:-data/visdrone_levelx/val}"   # levelx split dir OR image root
SEQ_GLOB="${2:-*}"                      # sequence-name filter

# --- model / run knobs (env-overridable) ------------------------------------
CKPT="${CKPT:-tmp/nomap-s-attn/ckpt-best}"
CONFIG="${CONFIG:-configs/visdrone_eval_5s.py}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8x_visdrone.pt}"
YOLO_CONF="${YOLO_CONF:-0.7}"
EGO_MOTION="${EGO_MOTION:-true}"
OUTDIR="${OUTDIR:-tmp/visdrone_e2e}"
DRY_RUN="${DRY_RUN:-0}"

# --- per-split image roots (levelx mode). Override via env. ------------------
TRAIN_IMG_ROOT="${TRAIN_IMG_ROOT:-VisDrone2019-MOT-train/sequences}"
TESTDEV_IMG_ROOT="${TESTDEV_IMG_ROOT:-VisDrone2019-MOT-test-dev/sequences}"
VAL_IMG_ROOT="${VAL_IMG_ROOT:-full-visdrone/sequences}"
IMG_ROOT="${IMG_ROOT:-}"               # if set, wins for ALL splits/seqs

cd "$PROJ" || { echo "FATAL: cannot cd $PROJ"; exit 1; }
mkdir -p "$OUTDIR"
shopt -s nullglob

# Echo the seq's frame dir (parent that holds frames directly OR in img/), or
# return non-zero if no frames are found for it.
seq_frame_dir() {  # $1=seq  $2=image_root
  local cand="$2/$1"
  if compgen -G "$cand/*.jpg" >/dev/null || compgen -G "$cand/*.jpeg" >/dev/null \
     || compgen -G "$cand/*.png" >/dev/null || compgen -G "$cand/img/*.jpg" >/dev/null; then
    echo "$cand"; return 0
  fi
  return 1
}

img_root_for_split() {  # $1=split
  if [[ -n "$IMG_ROOT" ]]; then echo "$IMG_ROOT"; return; fi
  case "$1" in
    train)    echo "$TRAIN_IMG_ROOT" ;;
    test-dev) echo "$TESTDEV_IMG_ROOT" ;;
    val)      echo "$VAL_IMG_ROOT" ;;
    *)        echo "" ;;
  esac
}

# --- build the work list as "seq::frame_dir" entries ------------------------
worklist=()
if compgen -G "$SRC/*.txt" >/dev/null; then
  MODE="levelx"
  split="$(basename "$SRC")"
  img_root="$(img_root_for_split "$split")"
  if [[ -z "$img_root" ]]; then
    echo "FATAL: unknown split '$split' (expected train/val/test-dev). "\
         "Set IMG_ROOT=<dir> to point at the frames explicitly." >&2
    exit 1
  fi
  echo "[e2e] mode=levelx  split=$split  img_root=$img_root"
  for f in "$SRC"/$SEQ_GLOB.txt; do
    seq="$(basename "$f" .txt)"
    if fdir="$(seq_frame_dir "$seq" "$img_root")"; then
      worklist+=("$seq::$fdir")
    else
      echo "[e2e] SKIP $seq: no frames under $img_root/$seq"
    fi
  done
else
  MODE="imgroot"
  echo "[e2e] mode=imgroot  root=$SRC"
  for d in "$SRC"/$SEQ_GLOB/; do
    seq="$(basename "$d")"
    if fdir="$(seq_frame_dir "$seq" "$SRC")"; then
      worklist+=("$seq::$fdir")
    else
      echo "[e2e] SKIP $seq: no frames in $d (or $d/img)"
    fi
  done
fi

if [[ ${#worklist[@]} -eq 0 ]]; then
  echo "[e2e] nothing to do: no sequences matched '$SEQ_GLOB' under $SRC" >&2
  exit 1
fi

echo "[e2e] ${#worklist[@]} sequence(s) to process; outputs -> $OUTDIR/"
if [[ "$DRY_RUN" == "1" ]]; then
  for entry in "${worklist[@]}"; do
    echo "  ${entry%%::*}  <-  ${entry#*::}"
  done
  echo "[e2e] DRY_RUN=1 — resolution only, no rendering."
  exit 0
fi

# --- run --------------------------------------------------------------------
n_ok=0; n_fail=0
for entry in "${worklist[@]}"; do
  seq="${entry%%::*}"; fdir="${entry#*::}"
  out="$OUTDIR/e2e_${seq}.mp4"
  echo "[e2e] $seq  <-  $fdir  ->  $out   ($(date '+%T'))"
  if $PY "$DRIVER" \
        --input-images "$fdir" \
        --anno-format  yolo-deepsort \
        --yolo-weights "$YOLO_WEIGHTS" \
        --yolo-conf    "$YOLO_CONF" \
        --map-model    none \
        --ckpt         "$CKPT" \
        --config       "$CONFIG" \
        --ego-motion   "$EGO_MOTION" \
        --output       "$out"; then
    n_ok=$((n_ok + 1))
  else
    echo "[e2e] FAILED on $seq"
    n_fail=$((n_fail + 1))
  fi
done

echo "[e2e] done: $n_ok ok, $n_fail failed.  Outputs in $OUTDIR/"
