# Cross-domain VisDrone eval — 2 s prediction horizon (10 steps @ 5 Hz).
#
# Inherits the EXACT config the no-map S-ATTN baseline (tmp/nomap-s-attn) was
# trained with (configs/levelx_train.py), then: (a) forces map_model=None so the
# checkpoint loads with no MapEncode submodule, (b) shortens the prediction
# horizon to what the short VisDrone clips can supply, (c) enables clustered
# min-of-K like configs/levelx_eval.py. VRUs stay in the neighbour set (no
# exclude_groups) exactly as in training — see docs note (g) in
# docs/mapencoder_and_training_brief.txt.
#
# Run (no --map_dir → no-map path):
#   $PY -m contextvae.main --test data/visdrone_levelx/val \
#       --config configs/visdrone_eval_2s.py --ckpt tmp/nomap-s-attn/ckpt-best
#
# The model's `horizon` is only a rollout-loop count (model.py:83), so the
# 25-step checkpoint loads cleanly into this 10-step model.
#
# For a raw-K sweep (K∈{1,5,10,20}) instead of clustered min-of-5, set
# clustering = 0 and pred_samples = K.
from .levelx_train import *

PRED_HORIZON = 10  # 2 s @ 5 Hz

test_dataloader["pred_horizon"] = PRED_HORIZON
model["horizon"] = PRED_HORIZON
model["map_model"] = None

clustering = 5 * pred_samples  # minADE_k / minFDE_k, matching configs/levelx_eval.py
