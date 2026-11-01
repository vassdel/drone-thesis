# levelX (inD+uniD+rounD) ContextVAE training — primary aerial-orthomap config.
# Horizons are in frames at 5 Hz (one frame = 0.2 s). The 6-D state's vx, vy are
# meters per timestep, not m/s (multiply by 5 to get m/s).
MIN_OB_HORIZON = 10   # minimum observed frames to form a window (2 s)
OB_HORIZON = 10       # observation window: 10 frames = 2 s
PRED_HORIZON = 25     # prediction horizon: 25 frames = 5 s
OB_RADIUS = 30        # neighbor-selection radius in meters
MAP_SIZE = 224        # orthomap patch side in pixels (~30 m coverage post-rotation)


lr = 3e-4
epochs = 60
test_since = 5        # start evaluating from epoch 5
preload_data = False
pred_samples = 5
clustering = 0        # eval-time clustering is enabled in levelx_eval.py

train_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["TARGET"],  # only vehicles tagged TARGET are predicted; VRUs stay as context
    batch_size=256,
    batches_per_epoch=1000, # ~25% of the ~1M-trajectory train split per epoch
    # Training-time map augmentation (color jitter, small rotation/scale).
    # Only active when a map is used (--map_dir present), so it is a no-op for
    # the no-map S-ATTN-only baseline.
    map_augment=True,
)
test_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["TARGET"],
    batch_size=512
)

model = dict(
    horizon = PRED_HORIZON,
    ob_radius = OB_RADIUS,
    hidden_dim = 512,
    # Aerial-orthomap M-ATTN variant: ResNet-18 backbone with ImageNet-pretrained
    # weights and map-input dropout (see contextvae/model.py). Set to None for the
    # no-map S-ATTN-only baseline.
    map_model = "resnet18"
)
