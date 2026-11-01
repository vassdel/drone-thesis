# levelX (inD+uniD+rounD) LSTM baseline (Ntousis-style predictor) — training.
from contextvae.lstm_baseline import LSTMBaseline

# Same observation/prediction window as levelx_train.py so the LSTM baseline
# consumes the exact same data.py tensors as the M-ATTN and no-map S-ATTN runs.
# Horizons are in frames at 5 Hz (one frame = 0.2 s); the 6-D state's vx, vy are
# meters per timestep, not m/s.
MIN_OB_HORIZON = 10   # minimum observed frames to form a window (2 s)
OB_HORIZON = 10       # observation window: 10 frames = 2 s
PRED_HORIZON = 25     # prediction horizon: 25 frames = 5 s
OB_RADIUS = 30        # neighbor-selection radius in meters
MAP_SIZE = 224        # orthomap patch side in pixels (unused by the LSTM)


lr = 3e-4
epochs = 40           # LSTM converges faster than the VAE (~1/3 of the M-ATTN budget)
test_since = 5
preload_data = False
pred_samples = 5      # tile-broadcast by LSTMBaseline.forward (unimodal)
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
    input_dim=10,  # Ntousis layout: (x, y) for [up, left, target, right, down]
)
