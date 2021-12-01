# Trainer entry point.

from . import cluster
from . import model
from . import param
from . import util

incomplete = {}
complete = []

def start():
    util.log('Starting trainer task.')

    if not model.load():
        model.generate()
        model.save()

    for dest in cluster.inferencers:
        cluster.comm.send((cluster.comm.rank, param.MSG_RELOAD), dest)

    util.log('Waiting for trajectory messages.')

    while True:
        sender, mtype = cluster.comm.recv()

        if mtype == param.MSG_INCOMPLETE_TRAJECTORY:
            observation, mcts, tag = cluster.comm.recv(source=sender)

            if tag not in incomplete:
                incomplete[tag] = []

            incomplete[tag].append((observation, mcts))

        if mtype == param.MSG_COMPLETE_TRAJECTORY:
            tag, value = cluster.comm.recv(source=sender)

            # Translate trajectories, apply value multiplier
            # We start with the observation immediately BEFORE
            # the terminal state is reached, so we flip the value
            # initially.
            value_mul = -1

            if tag not in incomplete:
                raise RuntimeError('tag {} not in incomplete trajectories'.format(tag))

            for (obs, mcts) in reversed(incomplete[tag]):
                complete.append((obs, mcts, value_mul * value))

                value_mul *= -1

            del incomplete[tag]

            # If complete trajectories passed iteration size,
            # perform training step now.

            if len(complete) >= param.TRAIN_GENSIZE:
                model.train(complete)
                
                for dest in cluster.inferencers:
                    cluster.comm.send((cluster.rank, param.MSG_RELOAD), dest)

                complete.clear()
