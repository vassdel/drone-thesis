# levelX (inD+uniD+rounD) evaluation — inherits the training config.
from .levelx_train import *

# minADE_k / minFDE_k: cluster the sampled trajectories into k=5*pred_samples
# modes before computing best-of-k error.
clustering = 5 * pred_samples
