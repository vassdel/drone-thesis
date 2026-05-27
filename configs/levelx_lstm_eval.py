from configs.levelx_lstm_train import *

# The LSTM is unimodal — LSTMBaseline.forward tile-broadcasts the deterministic
# output K times when n_predictions > 0, so KMeans-over-K samples sees identical
# points and trips ConvergenceWarning while doing no useful work. Skip clustering
# entirely; minADE_K and minFDE_K collapse to ADE_d / FDE_d by construction.
clustering = 0
