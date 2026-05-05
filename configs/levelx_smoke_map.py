# Smoke-test config — orthomap M-ATTN variant with ResNet-18 backbone.
# Inherits levelx_train.py and overrides `model.map_model` to wire in the
# torchvision resnet18 backbone (see contextvae/model.py:97-98). Use this
# smoke config to exercise the full map pipeline before the Wednesday
# orthomap M-ATTN training run.
#
# Run from the repo root in the `new_xupei_env` conda env:
#
#   python -m contextvae.main \
#       --train data/levelx/train \
#       --test  data/levelx/val \
#       --map_dir data/levelx/map \
#       --config configs/levelx_smoke_map.py \
#       --ckpt tmp/smoke_map
#
# For the literal Tuesday-checklist smoke (1 epoch on a 2-recording subset),
# symlink two `*.txt`/`*.info` pairs from data/levelx/train into a temp dir
# and pass that as `--train`. `--map_dir` MUST be supplied — the map path is
# active here and the loader needs the matching `<scene>.pkl` files.
from .levelx_train import *

epochs = 1
test_since = 1

# Copy the base `model` dict before mutating so we don't poison other configs
# that might import levelx_train in the same Python session.
model = dict(model)
model["map_model"] = "res18"
