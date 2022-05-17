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
MODEL_RESIDUALS = 1 # residual layer count
MODEL_FILTERS = 4   # filters per convolutional/half-residual layer

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
TRAIN_EPOCHS = 1      # training epochs
TRAIN_LR = 0.001      # learning rate (SGD)

TRAIN_ALPHA_INIT = 2
TRAIN_ALPHA_DECAY = 0.9
TRAIN_ALPHA_CUTOFF = 15
TRAIN_ALPHA_FINAL = 0.0001

# MCTS parameters

MCTS_CPUCT = 1               # c_puct parameter (see PUCT formula in alphazero)
MCTS_NODES = 512             # target root node count per move
MCTS_NOISE_ALPHA = 1 / PSIZE # Dirichlet noise alpha
MCTS_NOISE_WEIGHT = 0.01     # noise weight per expansion

FORCE_SELECT_UNVISITED_CHILDREN = False # always select an unvisisted child when selecting() nodes to expand

# Eval parameters

EVAL_BATCH_SIZE = 32 # concurrent eval games
EVAL_GAMES = 32      # total eval games
EVAL_INTERVAL = 25  # evalute model every n generations
