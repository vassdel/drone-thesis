# levelX (inD+uniD+rounD) — short pre-flight of the map-on full-data run.
# Inherits levelx_train (map_model="resnet18", map_augment=True, batch_size=256)
# and reduces epochs/batches so the run finishes quickly, surfacing OOM,
# dataloader bottlenecks, posterior collapse, or eval regressions before the
# multi-hour headline run.
from .levelx_train import *

epochs = 5
test_since = 0  # evaluate every epoch, starting at epoch 0

# Shallow-copy so we don't mutate the parent module's dict.
train_dataloader = {**train_dataloader, "batches_per_epoch": 200}
