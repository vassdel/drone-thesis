MIN_OB_HORIZON = 10
OB_HORIZON = 10
PRED_HORIZON = 25
OB_RADIUS = 30
MAP_SIZE = 224


lr = 3e-4
epochs = 100
test_since = 60
preload_data = True
pred_samples = 5
clustering = 0

train_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["TARGET"],
    batch_size=128,
    batches_per_epoch=200 # one epoch
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
    map_model = None
)
