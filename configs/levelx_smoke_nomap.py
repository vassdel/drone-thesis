# Smoke-test config — no-map S-ATTN-only baseline.
# Inherits everything from levelx_train.py (which already sets map_model=None),
# then trims epochs to 1 so a 2-recording smoke run finishes in minutes.
#
# Run from the repo root in the `new_xupei_env` conda env:
#
#   python -m contextvae.main \
#       --train data/levelx/train \
#       --test  data/levelx/val \
#       --map_dir data/levelx/map \
#       --config configs/levelx_smoke_nomap.py \
#       --ckpt tmp/smoke_nomap
from .levelx_train import *

epochs = 1
test_since = 1
