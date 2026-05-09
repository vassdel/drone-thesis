# Disposable 1-epoch smoke-test config. Inherits everything from levelx_train,
# overrides only the duration and eval-gate so a single recording finishes in
# ~1-2 min on A40 and a checkpoint is produced for the eval-determinism step.
from .levelx_train import *

epochs = 1
test_since = 0  # run eval at epoch 0 so we get a checkpoint with eval metrics

# Shallow-copy the dict so we don't mutate the parent module's dict object.
train_dataloader = {**train_dataloader, "batches_per_epoch": 20}
