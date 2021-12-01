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

    for i in range(param.EVAL_GAMES):
        game = param.SELECTED_ENV()
        turn = random.randint(0, 1)
        result = None
        
        while True:
            result = game.terminal()

            if result is not None:
                break

            t = mcts.Tree()

            for a in game.actions:
                t.advance(a)

            if str(t.env) != str(game):
                raise RuntimeError('game diverged: g:\n{}\ntree:\n{}'.format(game, t.env))

            if turn == 1:
                while t.select() is not None:
                    policy = np.random.dirichlet([1.0] * param.PSIZE)
                    value = np.random.randint(-100, 100) / 100

                    t.expand(policy, value)
            else:
                while t.select() is not None:
                    policy, value = model.infer([t.env.observe()])
                    t.expand(policy[0], value[0])

            #print('rng n: {}'.format(tree_rng.root.n))
            #print('net n: {}'.format(tree_net.root.n))
            #print('rng children: {}'.format(tree_rng.root.children))
            #print('net children: {}'.format(tree_net.root.children))

            # Make the next move.
            action = t.pick()

            game.push(action)

            if i == 0:
                print('eval game 0 state: \n{}'.format(game))

            turn = 1 - turn
        
        # Check game result

        if turn == 0:
            # Result as-is
            result = (result + 1) / 2
            pass
        else:
            # Invert result, RNG to move
            result = (1 - result) / 2

        if i == 0:
            print('eval game 0 result: {}'.format(result))

        util.log('Finished evaluation game with score: {}'.format(result))
        score += result

    util.log('Finished evaluation, performance {}'.format(score / param.EVAL_GAMES))
    return score / param.EVAL_GAMES

        

