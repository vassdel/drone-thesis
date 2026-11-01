# Waymo evaluation — inherits the training config.
from .waymo_train import *
test_dataloader["traj_max_overlap"] = None      # use all sub-trajectories at eval time
test_dataloader["inclusive_groups"] = ["VEHICLE"]  # evaluate on VEHICLE agents only
clustering = 5*pred_samples  # cluster samples into k modes for minADE_k / minFDE_k
