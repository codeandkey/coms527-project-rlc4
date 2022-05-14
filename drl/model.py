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

cuda_available = torch.cuda.is_available()

if not cuda_available:
    util.log('WARNING: CUDA is not available')

class DRLConvolutional(nn.Module):
    def __init__(self, features = param.FEATURES):
        super().__init__()

        self.conv1 = nn.Conv2d(features, param.MODEL_FILTERS, (3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(param.MODEL_FILTERS)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = x.view(-1, param.FEATURES, param.WIDTH, param.HEIGHT)
        x = x.permute(0, 3, 1, 2)
        return self.relu(self.bn1(self.conv1(x)))

class DRLResidual(nn.Module):
    def __init__(self, filters = param.MODEL_FILTERS):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)

        self.conv1 = nn.Conv2d(filters, filters, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(filters, filters, (3, 3), padding=(1, 1), bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += skip
        return self.relu(x)

class DRLModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        # Convolutional block
        self.conv = DRLConvolutional()

        # Residual blocks
        for i in range(param.MODEL_RESIDUALS):
            setattr(self, 'residual_{}'.format(i), DRLResidual())

        # Policy head layers
        self.pconv1 = nn.Conv2d(param.MODEL_FILTERS, 32, (1, 1), stride=1)
        self.pbn = nn.BatchNorm2d(32)
        self.pfc1 = nn.Linear(32 * param.WIDTH * param.HEIGHT, 7)
        self.pout = nn.LogSoftmax(dim=1)

        # Value head layers
        self.vconv1 = nn.Conv2d(param.MODEL_FILTERS, 3, (1, 1), stride=1)
        self.vbn = nn.BatchNorm2d(3)
        self.vfc1 = nn.Linear(3 * param.WIDTH * param.HEIGHT, 128)
        self.vfc2 = nn.Linear(128, 1)

    def forward(self, observation):
        x = observation

        # Convolutional layer
        x = self.conv(x)

        # Residuals
        for i in range(param.MODEL_RESIDUALS):
            x = getattr(self, 'residual_{}'.format(i))(x)

        # Policy head
        ph = self.relu(self.pbn(self.pconv1(x)))
        #ph = ph.view(-1, 32 * param.WIDTH * param.HEIGHT)
        ph = ph.flatten(1)
        ph = self.pfc1(ph)
        ph = self.pout(ph).exp()

        # Value head
        vh = self.relu(self.vbn(self.vconv1(x)))
        #vh = vh.view(-1, 3 * param.WIDTH * param.HEIGHT)
        vh = vh.flatten(1)
        vh = self.relu(self.vfc1(vh))
        vh = self.vfc2(vh)
        vh = torch.tanh(vh)

        return ph, vh

def load():
    """Loads the model from the disk."""
    global loaded

    try:
        loaded = torch.jit.load(param.MODEL_PATH)

        if cuda_available:
            loaded = loaded.cuda()

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

    tensor = torch.tensor(observations, dtype=torch.float32)

    if cuda_available:
        tensor = tensor.cuda()

    policy, value = loaded.forward(tensor)

    return policy.cpu().detach().numpy(), value.cpu().detach().numpy()

def generate():
    """Generates a new model."""
    global loaded

    util.log('Generating model')

    mod = DRLModule()
    inputs = torch.tensor(np.zeros((1, param.WIDTH, param.HEIGHT, param.FEATURES), dtype=np.float32))

    loaded = torch.jit.trace(mod, inputs)

    if cuda_available:
        loaded = loaded.cuda()

def train(trajectories):
    """Trains the loaded model on a collection of trajectories."""
    util.log('Training model on {} trajectories'.format(len(trajectories)))

    loaded.train(True)

    def lossfn(policy, value, mcts, result):
        value_loss = (value - result) ** 2
        policy_loss = torch.sum(-mcts * (policy.float() + 1e-5).float().log(), 1)

        return (value_loss.view(-1).float() + policy_loss).mean()

    optimizer = optim.SGD(loaded.parameters(), lr=param.TRAIN_LR, momentum=0.9)

    trajectories = [[
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor(mcts, dtype=torch.float32),
        torch.tensor(result, dtype=torch.float32)
    ] for (obs, mcts, result) in trajectories]

    if cuda_available:
        for t in trajectories:
            for j in t:
                j.cuda()

    loader = data_utils.DataLoader(
        trajectories,
        batch_size=param.TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    avgloss = 0
    count = 0

    for epoch in range(param.TRAIN_EPOCHS):
        for i, (obs, mcts, result) in enumerate(loader, 0):
            optimizer.zero_grad()

            policy, value = loaded(obs.cuda())

            value = torch.squeeze(value)

            loss = lossfn(policy, value, mcts.cuda(), result.cuda())
            loss.backward()

            optimizer.step()

            closs = loss.cpu().item()
            avgloss += closs

            if i % 10 == 9:
                util.log('Epoch {}/{}, batch {}/{}, loss {:.1f}        \r'.format(
                    epoch + 1,
                    param.TRAIN_EPOCHS,
                    i + 1,
                    int(len(trajectories) / param.TRAIN_BATCH_SIZE),
                    closs
                ))

            count += mcts.shape[0]

    avgloss /= count

    print()
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
    loaded.train(False)
    return gen

def generation():
    """Returns the current model generation, or 0 if it has not been saved."""
    gen = 0
    if path.exists('generation'):
        with open('generation', 'r') as f:
            gen = int(f.read())

    return gen
