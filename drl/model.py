# Model management

from . import param
from . import util

import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
import torch.utils.data as data_utils

import json

from os import path

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
        ph = torch.flatten(x, 1)
        ph = self.pfc1(ph)
        ph = self.prelu1(ph)
        ph = self.pfc2(ph)
        ph = self.pout(ph)

        # Value head
        vh = torch.flatten(x, 1)
        vh = self.vfc1(vh)
        vh = self.vrelu1(vh)
        vh = self.vfc2(vh)
        vh = self.vout(vh)

        return ph, vh

def load():
    """Loads the model from the disk."""
    global loaded

    try:
        loaded = torch.jit.load(param.MODEL_PATH).cuda()
        util.log('Loaded model from {}'.format(param.MODEL_PATH))
    except ValueError:
        return False

    return True

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

def train(trajectories):
    """Trains the loaded model on a collection of trajectories."""
    util.log('Training model on {} trajectories'.format(len(trajectories)))

    def lossfn(policy, value, mcts, result):
        return nn.MSELoss()(value, result) -torch.sum(torch.log(policy + 0.001) * mcts)

    optimizer = optim.SGD(loaded.parameters(), lr=param.TRAIN_LR, momentum=0.9)

    trajectories = [[
        torch.tensor(obs, dtype=torch.float32).cuda(),
        torch.tensor(mcts, dtype=torch.float32).cuda(),
        torch.tensor(result, dtype=torch.float32).cuda()
    ] for (obs, mcts, result) in trajectories]

    loader = data_utils.DataLoader(
        trajectories,
        batch_size=param.TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    avgloss = 0
    count = 0

    for epoch in range(param.TRAIN_EPOCHS):
        closs = 0

        for i, (obs, mcts, result) in enumerate(loader, 0):
            optimizer.zero_grad()

            policy, value = loaded(obs)

            value = torch.squeeze(value)

            loss = lossfn(policy, value, mcts, result)
            loss.backward()

            optimizer.step()

            closs += loss.cpu().item()
            avgloss += loss.cpu().item()

            if i % 10 == 9:
                util.log('Epoch {}/{}, batch {}/{}, loss {:.1f}'.format(
                    epoch + 1,
                    param.TRAIN_EPOCHS,
                    i + 1,
                    int(len(trajectories) / param.TRAIN_BATCH_SIZE),
                    closs
                ))

                closs = 0

            count += mcts.shape[0]

    avgloss /= count

    util.log('Finished training. Average loss by input: {}'.format(avgloss))
    save()

    gen = generation()
    gen += 1

    with open('generation', 'w') as f:
        f.write(str(gen))

    lossbuf = []

    if path.exists('loss'):
        with open('loss', 'r') as f:
            lossbuf = json.load(f)
    
    lossbuf.append([gen, avgloss])

    with open('loss', 'w') as f:
        json.dump(lossbuf, f)

    util.log('Wrote model generation {}'.format(gen))
    return gen

def generation():
    gen = 0
    if path.exists('generation'):
        with open('generation', 'r') as f:
            gen = int(f.read())

    return gen