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

# Message types

MSG_RELOAD = 0
MSG_INCOMPLETE_TRAJECTORY = 1
MSG_COMPLETE_TRAJECTORY = 2
MSG_INFERENCE = 3
MSG_PAUSE = 4
MSG_UNPAUSE = 5

# Model parameters

TRAIN_GENSIZE = 256
TRAIN_BATCH_SIZE = 4
TRAIN_EPOCHS = 2

# MCTS parameters

MCTS_CPUCT = 2
MCTS_NODES = 32
MCTS_NOISE_ALPHA = 1 / PSIZE
MCTS_NOISE_WEIGHT = 0.05

# Eval parameters

EVAL_GAMES = 25
EVAL_INTERVAL = 10