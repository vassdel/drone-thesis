# Disposable 5-epoch pre-flight config for the map-on full-data run.
# Inherits everything from levelx_train (map_model="resnet18", map_augment=True,
# batch_size=256), reduces epochs/batches so the pre-flight finishes in ~20 min
# on A40 and surfaces OOM, dataloader bottlenecks, posterior collapse, or eval
# regressions before the 4-5 hour headline run.
from .levelx_train import *

epochs = 5
test_since = 0  # eval every epoch starting at epoch 0

# Shallow-copy so we don't mutate the parent module's dict.
train_dataloader = {**train_dataloader, "batches_per_epoch": 200}
