# levelX smoke test — no-map S-ATTN-only baseline.
# Inherits levelx_train.py (which sets map_model=None) and trims epochs to 1 so
# a small smoke run finishes in minutes.
#
# Run from the repo root:
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
