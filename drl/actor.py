# Actor routines.

from . import cluster
from . import mcts
from . import param
from . import util

import numpy as np
import random
import uuid

trees = [mcts.Tree() for i in range(param.ENVS_PER_ACTOR)]
tags = [str(uuid.uuid4()) for i in range(param.ENVS_PER_ACTOR)]

def start():
    """Starts an actor node."""
    # Start actor loop

    util.log('Starting actor loop')
    completed = 0

    while True:
        next_batch = np.empty((param.ENVS_PER_ACTOR, param.WIDTH, param.HEIGHT, param.FEATURES))

        for i in range(len(trees)):
            nxt = trees[i].select()

            while nxt is None:
                # Send incomplete trajectory to trainer
                snapshot = trees[i].snapshot()
                trajectory = (trees[i].env.observe(), snapshot, tags[i])

                cluster.comm.send((cluster.rank, param.MSG_INCOMPLETE_TRAJECTORY), cluster.trainer)
                cluster.comm.send(trajectory, cluster.trainer)

                if cluster.rank == cluster.actors[0] and i == 0:
                    print('====== making move ======')
                    print('snapshot\n{}'.format(' '.join(map(lambda s: str(round(s * trees[i].root.n)), snapshot))))
                    print('state\n{}'.format(trees[i].env),end=None)

                    levels = ['']
                    levels = [' ', '░ ', '▒', '▓', '█']

                    def level(n):
                        return min(int(n * len(levels)), len(levels) - 1)

                    print(' ' + ''.join([levels[level(snapshot[i])] for i in range(7)]))
                   
                # Advance environment immediately
                action = trees[i].pick()

                if cluster.rank == cluster.actors[0] and i == 0:
                    print('picking {}'.format(action))
                
                # Check for terminal state
                tvalue = trees[i].terminal()

                if tvalue is not None:
                    # DON'T Send final observation to training
                    # As there is no policy at this point
                    # cluster.comm.send((cluster.rank, param.MSG_INCOMPLETE_TRAJECTORY), cluster.trainer)
                    # cluster.comm.send((environments[i].observe(), tags[i]), cluster.trainer)

                    # Send complete trajectory to trainer
                    cluster.comm.send((cluster.rank, param.MSG_COMPLETE_TRAJECTORY), cluster.trainer)
                    cluster.comm.send((tags[i], tvalue), cluster.trainer)
                
                    # Replace environment and tag
                    trees[i] = mcts.Tree()
                    tags[i] = str(uuid.uuid4())

                    completed += 1

                    if completed % 20 == 0:
                        util.log('{} trajectories completed'.format(completed))

                # Re-select
                nxt = trees[i].select()

            # Insert observation into batch
            next_batch[i] = nxt

        # Send batch to random inference
        target = random.choice(cluster.inferencers)

        cluster.comm.send((cluster.rank, param.MSG_INFERENCE), target)
        cluster.comm.send(next_batch, target)

        # Wait for policy/value response
        policy, value = cluster.comm.recv(source=target)

        # Perform expansion
        for i in range(len(trees)):
            trees[i].expand(policy[i], value[i])
