# nuScenes evaluation — inherits the training config.
from .nuscenes_train import *
test_dataloader["batch_size"] = 32
clustering = 2000  # cluster samples into 2000 modes for minADE_k / minFDE_k
