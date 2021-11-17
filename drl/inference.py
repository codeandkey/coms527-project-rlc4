# Inference node type

from . import cluster
from . import model
from . import param
from . import util

import numpy as np

def start():
    util.log('Starting inference task.')

    # Wait for first model reload from trainer
    sender, mtype = cluster.comm.recv(source=cluster.trainer)

    if sender != cluster.trainer or mtype != param.MSG_RELOAD:
        raise RuntimeError('Invalid initial message')

    # Load model
    model.load()

    # Wait for observations / reloads
    while True:
        sender, mtype = cluster.comm.recv()

        if sender == cluster.trainer:
            if mtype != param.MSG_RELOAD:
                raise RuntimeError('Invalid message from trainer')

            model.load()
        else:
            if mtype != param.MSG_INFERENCE:
                raise RuntimeError('Invalid message from trainer')

            # Read observations
            observations = cluster.comm.recv(source=sender)

            # Perform inference
            policy, value = model.infer(observations)

            # Write back results
            cluster.comm.send((policy, value), sender)
