# Waymo ContextVAE training (distributed across 8 ranks).
OB_HORIZON = 6        # observation window in frames
PRED_HORIZON = 15     # prediction horizon in frames
OB_RADIUS = 30        # neighbor-selection radius in meters
MAP_SIZE = 224        # HD-map raster patch side in pixels


lr = 3e-4
epochs = 400
test_since = 120      # start evaluating from epoch 120
preload_data = False
pred_samples = 5
clustering = 0        # eval-time clustering is enabled in waymo_eval.py

train_dataloader = dict(
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["EGO", "VEHICLE"],
    batch_size=512,
    batches_per_epoch=500
)
test_dataloader = dict(
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["EGO", "VEHICLE"],
    # Cap overlap between consecutive sub-trajectories at 7 frames to shrink the
    # test set evaluated during training.
    traj_max_overlap=7,
    batch_size=256
)

model = dict(
    horizon = PRED_HORIZON,
    ob_radius = OB_RADIUS,
    hidden_dim = 512,
    map_model = "mobile2"
)
