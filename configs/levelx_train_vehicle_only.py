# levelX ContextVAE training — vehicle-only ablation (VRUs excluded entirely).
# Horizons are in frames at 5 Hz (one frame = 0.2 s).
MIN_OB_HORIZON = 10   # minimum observed frames to form a window (2 s)
OB_HORIZON = 10       # observation window: 10 frames = 2 s
PRED_HORIZON = 25     # prediction horizon: 25 frames = 5 s
OB_RADIUS = 30        # neighbor-selection radius in meters
MAP_SIZE = 224        # orthomap patch side in pixels


lr = 3e-4
epochs = 100
test_since = 60       # start evaluating from epoch 60
preload_data = True
pred_samples = 5
clustering = 0

# Vehicle-only ablation: drop VRU rows at load time so they appear in neither
# ego nor neighbor selection. Tests whether removing the large VRU neighbor
# share helps the baseline's social attention.
train_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["TARGET"],  # only TARGET vehicles are predicted
    exclude_groups=["VRU"],       # drop VRUs from the scene entirely
    batch_size=128,
    batches_per_epoch=200
)
test_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["TARGET"],
    exclude_groups=["VRU"],
    batch_size=512
)

model = dict(
    horizon = PRED_HORIZON,
    ob_radius = OB_RADIUS,
    hidden_dim = 512,
    map_model = None
)
