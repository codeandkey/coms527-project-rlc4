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

ENVS_PER_ACTOR = 32 # environments on each actor, also inference batchsize

# Model architecture
MODEL_RESIDUALS = 4 # residual layer count
MODEL_FILTERS = 64  # filters per convolutional/half-residual layer

# Message types

MSG_RELOAD = 0
MSG_INCOMPLETE_TRAJECTORY = 1
MSG_COMPLETE_TRAJECTORY = 2
MSG_INFERENCE = 3
MSG_PAUSE = 4
MSG_UNPAUSE = 5

# Model parameters

TRAIN_GENSIZE = 256   # complete trajectories per generation
TRAIN_BATCH_SIZE = 8  # training batchsize
TRAIN_EPOCHS = 5      # training epochs
TRAIN_LR = 0.003      # learning rate (SGD)

# MCTS parameters

MCTS_CPUCT = 1               # c_puct parameter (see PUCT formula in alphazero)
MCTS_NODES = 32              # target root node count per move
MCTS_NOISE_ALPHA = 1 / PSIZE # Dirichlet noise alpha
MCTS_NOISE_WEIGHT = 0.05     # noise weight per expansion

# Eval parameters

EVAL_BATCH_SIZE = 32 # concurrent eval games
EVAL_GAMES = 32      # total eval games
EVAL_INTERVAL = 10   # evalute model every n generations
