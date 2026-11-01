# Lyft evaluation — inherits the training config.
from .lyft_train import *
test_dataloader["traj_max_overlap"] = None      # use all sub-trajectories at eval time
test_dataloader["inclusive_groups"] = ["VEHICLE"]  # evaluate on VEHICLE agents only

