# Cross-domain VisDrone eval — 5 s prediction horizon (25 steps @ 5 Hz).
# This matches the native ContextVAE / inD-uniD horizon, so it is the most
# directly comparable to the in-distribution val numbers — but only the LONGER
# VisDrone clips supply a full 2 s + 5 s window (35 steps @ 5 Hz = 175 frames at
# 30 fps); short sequences contribute no windows here and are dropped by the
# loader. See configs/visdrone_eval_2s.py for the full rationale.
from .levelx_train import *

PRED_HORIZON = 25  # 5 s @ 5 Hz (native horizon)

test_dataloader["pred_horizon"] = PRED_HORIZON
model["horizon"] = PRED_HORIZON
model["map_model"] = None

clustering = 5 * pred_samples  # minADE_k / minFDE_k, matching configs/levelx_eval.py
