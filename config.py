"""
Central configuration for all hyperparameters.
Adjust values here — no need to touch any other file.
"""

BATCH_SIZE       = 256
PRETRAIN_EPOCHS  = 20       # AE pretraining before clustering
CLUSTER_EPOCHS   = 80       # deep clustering phase
RESET_INTERVAL   = 10       # reclustering / BRB every N epochs
ALPHA            = 0.7      # soft-reset interpolation (1 = no reset)

INPUT_DIM        = 784      # flattened MNIST
EMBEDDING_DIM    = 10
N_CLUSTERS       = 10
LEARNING_RATE    = 1e-3
CLUSTER_LOSS_WT  = 0.1      # weight of clustering loss in joint objective

PLOT_DIR         = "plots"
