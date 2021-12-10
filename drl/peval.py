# Routines for evaluating model performance

from . import param
from . import model
from . import mcts
from . import util

import random
import numpy as np

def evaluate():
    """Performs a model evaluation, returns the final score."""

    score = 0
    util.log('Starting evaluation over {} games'.format(param.EVAL_GAMES))

    games = [mcts.Tree() for i in range(param.EVAL_BATCH_SIZE)]
    mturn = [random.randint(0, 1) * 2 - 1 for i in range(param.EVAL_BATCH_SIZE)]
    results = []

    next_batch = np.empty((param.EVAL_BATCH_SIZE, param.WIDTH, param.HEIGHT, param.FEATURES))

    def advance_rng(ind):
        if games[ind].env.turn == mturn[ind]:
            raise RuntimeError('advance_rng() called on wrong turn')

        while games[ind].select() is not None:
            policy = np.random.dirichlet([param.MCTS_NOISE_ALPHA] * param.PSIZE)
            value = np.random.randint(-100, 100) / 100
        
            games[ind].expand(policy, value)

        #mv = None
        #while True:
        #    mv = random.randint(0, 6)
        #    if games[ind].env.lmm()[mv] > 0.5:
        #        break

        games[ind].pick()
        #games[ind].advance(mv)
        games[ind].clear_subtree()
    
    completed = 0

    while completed < param.EVAL_GAMES:
        # Build next batch.
        for i in range(len(games)):
            # If computer's turn, advance immediately
            if games[i].env.turn != mturn[i]:
                advance_rng(i)

            if games[i].terminal() is not None:
                value = games[i].terminal()
                results.append(value * mturn[i])
                completed += 1
                util.log('Current performance {:.2f}'.format(((sum(results) / len(results)) + 1) / 2))

                if completed % 20 == 0:
                    util.log('{} evaluations completed'.format(completed))

                games[i] = mcts.Tree()
                mturn[i] = random.randint(0, 1) * 2 - 1

                if games[i].env.turn != mturn[i]:
                    advance_rng(i)

            next_obs = games[i].select()

            while next_obs is None:
                # Advance environment immediately
                action = games[i].pick()
                games[i].clear_subtree()

                # Check for terminal state
                tvalue = games[i].terminal()

                if tvalue is not None:
                    results.append(tvalue * mturn[i])
                    completed += 1
                    util.log('Current performance {:.2f}'.format(((sum(results) / len(results)) + 1) / 2))

                    # Replace environment
                    games[i] = mcts.Tree()
                    mturn[i] = random.randint(0, 1) * 2 - 1

                    if games[i].env.turn != mturn[i]:
                        advance_rng(i)

                    completed += 1

                    if completed % 20 == 0:
                        util.log('{} evaluations completed'.format(completed))

                # Re-select
                next_obs = games[i].select()

            # Insert observation into batch
            next_batch[i] = next_obs

        # Run batch
        policy, value = model.infer(next_batch)

        # Perform expansion
        for i in range(len(games)):
            games[i].expand(policy[i], value[i]) 

    score = ((sum(results) / len(results)) + 1) / 2

    util.log('Finished evaluation, performance {}'.format(score))
    return score
