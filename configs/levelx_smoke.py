# levelX 1-epoch smoke test. Inherits levelx_train and overrides only the
# duration and eval gate so a single recording finishes quickly and produces a
# checkpoint for the eval-determinism check.
from .levelx_train import *

epochs = 1
test_since = 0  # evaluate at epoch 0 so the checkpoint carries eval metrics

# Shallow-copy so we don't mutate the parent module's dict.
train_dataloader = {**train_dataloader, "batches_per_epoch": 20}
