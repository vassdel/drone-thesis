#!/usr/bin/env bash
# ============================================================================
# run_visdrone_eval.sh
# Preprocess VisDrone-MOT splits -> ContextVAE .txt, then run the per-sequence
# cross-domain eval (micro + macro) for each split x horizon, ONE AT A TIME.
#
# Usage:
#   bash run_visdrone_eval.sh                     # uses the default roots below
#   bash run_visdrone_eval.sh <testdev_root> <train_root>
#   nohup bash run_visdrone_eval.sh > tmp/visdrone_eval/run_all.log 2>&1 &   # detached
#
# Each split root must be in official VisDrone-MOT layout:
#   <root>/annotations/<seq>.txt   (required)
#   <root>/sequences/<seq>/...     (optional — images only used for the cosmetic origin)
# ============================================================================
set -uo pipefail

# --- EDIT THESE to point at your downloaded split roots (or pass as args) ---
TESTDEV_ROOT="${1:-visdrone_raw/VisDrone2019-MOT-test-dev}"
TRAIN_ROOT="${2:-visdrone_raw/VisDrone2019-MOT-train}"
# ---------------------------------------------------------------------------

PROJ=/home/vdelis/ContextVAE
PY=/home/vdelis/anaconda3/envs/new_xupei_env/bin/python
CKPT=tmp/nomap-s-attn/ckpt-best
OUTROOT=data/visdrone_levelx
LOGDIR=tmp/visdrone_eval
HORIZONS=(2s 3s 5s)

cd "$PROJ" || { echo "FATAL: cannot cd $PROJ"; exit 1; }
mkdir -p "$LOGDIR"

# split label -> root path (parallel arrays; bash 4+ assoc would also work)
SPLITS=(test-dev train)
ROOTS=("$TESTDEV_ROOT" "$TRAIN_ROOT")

preprocess() {  # $1=root  $2=split
  local root="$1" split="$2"
  if [[ ! -d "$root/annotations" ]]; then
    echo "[preprocess] SKIP $split: no '$root/annotations' dir"
    return 1
  fi
  echo "[preprocess] $split  <-  $root   ($(date '+%T'))"
  $PY preprocessing/process_visdrone.py \
      --root "$root" --out "$OUTROOT/$split" --fps 30 --target-hz 5
  return $?
}

evaluate() {  # $1=split
  local split="$1"
  if [[ ! -d "$OUTROOT/$split" ]] || ! ls "$OUTROOT/$split"/*.txt >/dev/null 2>&1; then
    echo "[eval] SKIP $split: no preprocessed .txt under $OUTROOT/$split"
    return 1
  fi
  for H in "${HORIZONS[@]}"; do
    local log="$LOGDIR/${split}_perseq_${H}.log"
    echo "[eval] $split $H  ->  $log   (start $(date '+%T'))"
    $PY -m contextvae.eval_perseq \
        --test "$OUTROOT/$split" \
        --config "configs/visdrone_eval_${H}.py" \
        --ckpt "$CKPT" > "$log" 2>&1
    echo "[eval] $split $H  done exit=$?   ($(date '+%T'))"
  done
}

echo "======================= PREPROCESS ======================="
for i in "${!SPLITS[@]}"; do
  preprocess "${ROOTS[$i]}" "${SPLITS[$i]}"
done

echo "==================== EVALUATE (sequential) ==============="
for split in "${SPLITS[@]}"; do
  evaluate "$split"
done

echo "========================= SUMMARY ========================"
for split in "${SPLITS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    log="$LOGDIR/${split}_perseq_${H}.log"
    [[ -f "$log" ]] || continue
    echo "----- $split $H -----"
    # Print the MICRO line and the MACRO block (the headline aggregates).
    awk '/^MICRO/{p=2} p>0{print; p--} /^MACRO/{m=7} m>0{print; m--}' "$log"
  done
done
echo "ALL DONE $(date '+%F %T')"
