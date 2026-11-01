# nuScenes ContextVAE training. Horizons are in frames at 2 Hz (one frame = 0.5 s).
MIN_OB_HORIZON = 2    # minimum observed frames to form a window
OB_HORIZON = 5        # observation window: 5 frames = 2 s
PRED_HORIZON = 12     # prediction horizon: 12 frames = 6 s
OB_RADIUS = 30        # neighbor-selection radius in meters
MAP_SIZE = 224        # HD-map raster patch side in pixels


lr = 3e-4
epochs = 100
test_since = 60       # start evaluating from epoch 60
preload_data = True
pred_samples = 5
clustering = 0        # eval-time clustering is enabled in nuscenes_eval.py

train_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["CHALLENGE"],
    batch_size=128,
    batches_per_epoch=200
)
test_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["CHALLENGE"],
    batch_size=512
)

model = dict(
    horizon = PRED_HORIZON,
    ob_radius = OB_RADIUS,
    hidden_dim = 512,
    map_model = "resnet18"
)
