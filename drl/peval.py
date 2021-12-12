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
    ids = list(range(param.EVAL_BATCH_SIZE))
    results = [None for i in range(param.EVAL_GAMES)]

    next_game_id = param.EVAL_BATCH_SIZE
    next_batch = np.empty((param.EVAL_BATCH_SIZE, param.WIDTH, param.HEIGHT, param.FEATURES))

    def advance_rng(ind):
        if games[ind].env.turn == mturn[ind]:
            raise RuntimeError('advance_rng() called on wrong turn')

        while games[ind].select() is not None:
            policy = np.random.dirichlet([param.MCTS_NOISE_ALPHA] * param.PSIZE)
            value = np.random.randint(-100, 100) / 300
        
            games[ind].expand(policy, value)

        #mv = None
        #while True:
        #    mv = random.randint(0, 6)
        #    if games[ind].env.lmm()[mv] > 0.5:
        #        break

        games[ind].pick()
        #games[ind].advance(mv)
        games[ind].clear_subtree()

    def performance():
        score = 0
        count = 0

        for r in results:
            if r is not None:
                score += r
                count += 1

        return ((score / count) * 2) - 1
    
    def complete_game(i):
        # Check terminal value
        value = games[i].terminal()

        if value is None:
            raise RuntimeError('complete() called on incomplete game')

        # Store result from model pov
        results[ids[i]] = value * mturn[i]

        # Reset environment
        games[i] = mcts.Tree()
        mturn[i] = random.randint(0, 1) * 2 - 1
        ids[i] = next_game_id
        next_game_id += 1

        if games[i].env.turn != mturn[i]:
            advance_rng(i)

        util.log('Current performance {:.2f}'.format(performance()))

    while None in results:
        # Build next batch.
        for i in range(len(games)):
            # If computer's turn, advance immediately
            if games[i].env.turn != mturn[i]:
                advance_rng(i)

            if games[i].terminal() is not None:
                complete_game(i)

            next_obs = games[i].select()

            while next_obs is None:
                # Advance environment immediately
                action = games[i].pick()
                games[i].clear_subtree()

                # Check for terminal state
                tvalue = games[i].terminal()

                if games[i].terminal() is not None:
                    complete_game(i)

                # Re-select
                next_obs = games[i].select()

            # Insert observation into batch
            next_batch[i] = next_obs

        # Run batch
        policy, value = model.infer(next_batch)

        # Perform expansion
        for i in range(len(games)):
            games[i].expand(policy[i], value[i]) 

    score = performance()    

    util.log('Finished evaluation, performance {}'.format(score))
    return score
