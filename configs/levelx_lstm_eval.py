# levelX LSTM baseline (Ntousis-style predictor) evaluation.
from configs.levelx_lstm_train import *

# The LSTM is unimodal: LSTMBaseline.forward tile-broadcasts the single
# deterministic output K times, so clustering over K identical samples is
# meaningless (and trips a KMeans convergence warning). Disable it; minADE_k and
# minFDE_k then collapse to the deterministic ADE / FDE by construction.
clustering = 0
