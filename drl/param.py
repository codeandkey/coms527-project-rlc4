# Program parameters

from .env import c4

# Environment parameters
SELECTED_ENV = c4.Connect4

WIDTH = SELECTED_ENV.width
HEIGHT = SELECTED_ENV.height
FEATURES = SELECTED_ENV.features
PSIZE = SELECTED_ENV.psize

# Config parameters

MODEL_PATH = 'model.pt'

ENVS_PER_ACTOR = 32

# Model architecture
MODEL_RESIDUALS = 3
MODEL_FILTERS = 128

# Message types

MSG_RELOAD = 0
MSG_INCOMPLETE_TRAJECTORY = 1
MSG_COMPLETE_TRAJECTORY = 2
MSG_INFERENCE = 3
MSG_PAUSE = 4
MSG_UNPAUSE = 5

# Model parameters

TRAIN_GENSIZE = 1024
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 3
TRAIN_LR = 0.01

# MCTS parameters

MCTS_CPUCT = 1
MCTS_NODES = 64
MCTS_NOISE_ALPHA = 1 / PSIZE
MCTS_NOISE_WEIGHT = 0.05

# Eval parameters

EVAL_BATCH_SIZE = 32
EVAL_GAMES = 128
EVAL_INTERVAL = 25
