# Routines for evaluating model performance

from . import param
from . import model
from . import mcts
from . import util

import random
import numpy as np

count = None
score = None

def evaluate():
    """Performs a model evaluation, returns the final score."""

    global count
    global score

    util.log('Starting evaluation over {} games'.format(param.EVAL_GAMES))

    games = [mcts.Tree() for i in range(param.EVAL_BATCH_SIZE)]
    mturn = [random.randint(0, 1) * 2 - 1 for i in range(param.EVAL_BATCH_SIZE)]

    next_batch = np.empty((param.EVAL_BATCH_SIZE, param.WIDTH, param.HEIGHT, param.FEATURES))

    count = 0
    score = 0

    def advance_rng(ind):
        if games[ind].env.turn == mturn[ind]:
            raise RuntimeError('advance_rng() called on wrong turn')

        while games[ind].select() is not None:
            policy = np.random.dirichlet([1 / param.PSIZE] * param.PSIZE)
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

    def complete_game(i):
        global count
        global score

        # Check terminal value
        value = games[i].terminal()

        if value is None:
            raise RuntimeError('complete() called on incomplete game')

        # Store result from model pov
        gscore = ((value * mturn[i]) + 1) / 2
        score += gscore
        count += 1

        rword = 'LOSS' if gscore == 0 else 'WIN ' if gscore == 1 else 'DRAW'
        player = 'X' if mturn[i] == 1 else 'O'

        # Reset environment
        games[i] = mcts.Tree()
        mturn[i] = random.randint(0, 1) * 2 - 1

        if games[i].env.turn != mturn[i]:
            advance_rng(i)

        util.log('{} as {}, performance {:.2f} ({} / {})'.format(rword, player, score / count, score, count))

    while count < param.EVAL_GAMES:
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

    util.log('Finished evaluation, performance {}'.format(score / count))
    return score / count
