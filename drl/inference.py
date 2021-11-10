# Inference node type

import cluster
import model
import param

import numpy as np

def start():
    # Wait for first model reload from trainer
    rbuf = np.empty((2,), dtype=np.int32)
    cluster.comm.recv(rbuf, cluster.trainer)

    if rbuf[0] != cluster.trainer or rbuf[1] != param.MSG_RELOAD:
        raise RuntimeError('Invalid initial message')

    # Load model
    model.load()

    # Wait for observations / reloads
    while True:
        cluster.comm.recv(rbuf)

        sender = rbuf[0]
        mtype = rbuf[1]

        if sender == cluster.trainer:
            if mtype != param.MSG_RELOAD:
                raise RuntimeError('Invalid message from trainer')

            model.load()
        else:
            # Read observations
            observations = np.empty((param.ENVS_PER_ACTOR, param.FEATURES, param.WIDTH, param.HEIGHT))
            tags = [None] * param.ENVS_PER_ACTOR

            cluster.comm.recv(observations, sender)
            cluster.comm.recv(tags, sender)

            # Perform inference
            policy, value = model.infer(observations)

            # Write back results
            cluster.comm.send((policy, value), sender)