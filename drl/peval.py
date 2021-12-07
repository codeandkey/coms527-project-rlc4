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

    # store list of running games

    # while there are unfinished games:
    # walk through games, try select
    # build batches
    # evaluate batches

    games = [mcts.Tree() for i in range(param.EVAL_BATCH_SIZE)]
    turns = [random.randint(0, 1) for i in range(param.EVAL_BATCH_SIZE)]
    results = []

    next_batch = np.empty((param.EVAL_BATCH_SIZE, param.FEATURES, param.WIDTH, param.HEIGHT))

    def advance_rng(ind):
        if turns[ind] != 1:
            raise RuntimeError('advance_rng() called on wrong turn')

        while games[ind].select() is not None:
            policy = np.random.dirichlet([param.MCTS_NOISE_ALPHA] * param.PSIZE)
            value = np.random.randint(-100, 100) / 500

            games[ind].expand(policy, value)

        games[ind].pick()
        turns[ind] = 1 - turns[ind]
    
    completed = 0

    while completed < param.EVAL_GAMES:
        # Build next batch.
        for i in range(len(games)):
            # If computer's turn, advance immediately
            if turns[i] == 1:
                advance_rng(i)
                turns[i] = 0

            if games[i].terminal() is not None:
                # CPU just moved. We want a positive value from model POV
                value = games[i].terminal()
                results.append(value)
                completed += 1
                util.log('Current performance {:.2f}'.format(((sum(results) / len(results)) + 1) / 2))

                games[i] = mcts.Tree()
                turns[i] = random.randint(0, 1)

                if turns[i] == 1:
                    advance_rng(i)
                    turns[i] = 0

            next_obs = games[i].select()

            while next_obs is None:
                # Advance environment immediately
                action = games[i].pick()
                turns[i] = 1 - turns[i]

                # Check for terminal state
                tvalue = games[i].terminal()

                if tvalue is not None:
                    # Model just moved. Apply negated result
                    value = -games[i].terminal()
                    results.append(value)
                    completed += 1
                    util.log('Current performance {:.2f}'.format(((sum(results) / len(results)) + 1) / 2))

                    # Replace environment
                    games[i] = mcts.Tree()
                    turns[i] = random.randint(0, 1)

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
