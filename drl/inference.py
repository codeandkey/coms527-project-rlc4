# Inference node type

from . import cluster
from . import model
from . import param
from . import util

import numpy as np

def start():
    """Starts an inference node."""
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
            if mtype == param.MSG_RELOAD:
                model.load()

            if mtype == param.MSG_PAUSE:
                util.log('Pausing inference')
                snd, msg = cluster.comm.recv(None, cluster.trainer)
                
                if snd != cluster.trainer or msg != param.MSG_UNPAUSE:
                    raise RuntimeError('Unexpected unpause format')

                util.log('Resuming inference')

        else:
            if mtype != param.MSG_INFERENCE:
                raise RuntimeError('Invalid message from actor')

            # Read observations
            observations = cluster.comm.recv(source=sender)

            # Perform inference
            policy, value = model.infer(observations)

            # Write back results
            cluster.comm.send((policy, value), sender)
