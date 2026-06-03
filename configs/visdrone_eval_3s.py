# Cross-domain VisDrone eval — 3 s prediction horizon (15 steps @ 5 Hz).
# See configs/visdrone_eval_2s.py for the full rationale; only PRED_HORIZON differs.
from .levelx_train import *

PRED_HORIZON = 15  # 3 s @ 5 Hz

test_dataloader["pred_horizon"] = PRED_HORIZON
model["horizon"] = PRED_HORIZON
model["map_model"] = None

clustering = 5 * pred_samples  # minADE_k / minFDE_k, matching configs/levelx_eval.py
