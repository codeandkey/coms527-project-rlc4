# Model management

import param
import util

import numpy as np
import torch
import torch.nn as nn

loaded = None

class DRLModule(nn.Module):
    def __init__(self):
        super().__init__()

        # Residual blocks
        self.conv1 = nn.Conv2d(param.FEATURES, param.FEATURES, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(param.FEATURES, param.FEATURES, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()

        # Policy head layers
        self.pfc1 = nn.Linear(param.FEATURES * param.WIDTH * param.HEIGHT, 128)
        self.prelu1 = nn.ReLU()
        self.pfc2 = nn.Linear(128, param.PSIZE)
        self.pout = nn.Softmax(-1)

        # Value head layers
        self.vfc1 = nn.Linear(param.FEATURES * param.WIDTH * param.HEIGHT, 128)
        self.vrelu1 = nn.ReLU()
        self.vfc2 = nn.Linear(128, 1)
        self.vout = nn.Tanh()

    def forward(self, observation):
        # Residual layers
        x = observation
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        # Policy head
        ph = torch.flatten(x)
        ph = self.pfc1(ph)
        ph = self.prelu1(ph)
        ph = self.pfc2(ph)
        ph = self.pout(ph)

        # Value head
        vh = torch.flatten(x)
        vh = self.vfc1(vh)
        vh = self.vrelu1(vh)
        vh = self.vfc2(vh)
        vh = self.vout(vh)

        return ph, vh

def load():
    """Loads the model from the disk."""
    global loaded

    loaded = torch.jit.load(param.MODEL_PATH).cuda()
    util.log('Loaded model from {}'.format(param.MODEL_PATH))

def save():
    """Saves the model to the disk."""

    if loaded is None:
        raise RuntimeError('No model loaded')

    torch.jit.save(loaded, param.MODEL_PATH)
    util.log('Saved model to {}'.format(param.MODEL_PATH))
        
def infer(observations):
    """Runs an input through the model to produce an output."""

    if loaded is None:
        raise RuntimeError('No model loaded')

    tensor = torch.tensor(observations, dtype=torch.float32).cuda()
    policy, value = loaded.forward(tensor)

    return policy.cpu().detach().numpy(), value.cpu().detach().numpy()

def generate():
    """Generates a new model."""
    global loaded

    util.log('Generating model')

    mod = DRLModule()
    inputs = torch.tensor(np.zeros((1, param.FEATURES, param.WIDTH, param.HEIGHT), dtype=np.float32))

    loaded = torch.jit.trace(mod, inputs).cuda()

def train(batch):
    """Trains the loaded model on a trajectory batch."""
    print('train(): stub')