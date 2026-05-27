from contextvae.lstm_baseline import LSTMBaseline

# Same observation/prediction window as levelx_train.py so the LSTM baseline
# eats the exact same data.py tensors as the M-ATTN and no-map S-ATTN runs.
# 6-D state's vx, vy are meters per timestep (0.2 s at 5 Hz), NOT m/s.
MIN_OB_HORIZON = 10
OB_HORIZON = 10
PRED_HORIZON = 25
OB_RADIUS = 30
MAP_SIZE = 224


lr = 3e-4
epochs = 40           # LSTM converges faster than the VAE; ~33% of the M-ATTN budget.
test_since = 5
preload_data = False
pred_samples = 5      # Tile-broadcast by LSTMBaseline.forward (unimodal).
clustering = 0

train_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["TARGET"],
    batch_size=256,
    batches_per_epoch=1000,
)
test_dataloader = dict(
    min_ob_horizon=MIN_OB_HORIZON,
    ob_horizon=OB_HORIZON,
    map_size=MAP_SIZE,
    pred_horizon=PRED_HORIZON,
    ob_radius=OB_RADIUS,
    inclusive_groups=["TARGET"],
    batch_size=512,
)

model_cls = LSTMBaseline
model = dict(
    horizon=PRED_HORIZON,
    hidden_dim=64,
    input_dim=10,  # Ntousis layout: [up, left, target, right, down] x (x, y)
)
