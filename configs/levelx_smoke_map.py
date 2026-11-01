# levelX smoke test — orthomap M-ATTN variant (ResNet-18 backbone).
# Inherits levelx_train.py and overrides model.map_model to wire in the
# torchvision resnet18 backbone. Exercises the full map pipeline (patch loading,
# encoder, M-ATTN) on a tiny run before the full orthomap training run.
#
# Run from the repo root:
#
#   python -m contextvae.main \
#       --train data/levelx/train \
#       --test  data/levelx/val \
#       --map_dir data/levelx/map \
#       --config configs/levelx_smoke_map.py \
#       --ckpt tmp/smoke_map
#
# For a minimal 1-epoch smoke on a 2-recording subset, symlink two *.txt/*.info
# pairs from data/levelx/train into a temp dir and pass it as --train. --map_dir
# MUST be supplied: the map path is active and the loader needs the matching
# <scene>.pkl files.
from .levelx_train import *

epochs = 1
test_since = 1

# Copy the base `model` dict before mutating so we don't affect other configs
# that import levelx_train in the same Python session.
model = dict(model)
model["map_model"] = "res18"
