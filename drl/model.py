# Model management

import param

import numpy as np
import torch

loaded = None

def load():
    """Loads the model from the disk."""
    print('load(): stub')

def infer(x, lmm):
    """Runs an input through the model to produce an output."""
    print('infer(): stub')
    return np.zeros(param.PSIZE), 0.0

def train(batch):
    """Trains the loaded model on a trajectory batch."""
    print('train(): stub')
