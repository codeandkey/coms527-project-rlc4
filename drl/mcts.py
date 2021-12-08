# MCTS tree/node types

from . import param

import math
import numpy as np
import random

class Node:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.children = None
        self.parent = None
        self.turn = None

    def q(self):
        if self.n == 0:
            return 0
        
        return self.w / self.n

    def select(self):
        if not self.children:
            return None

        best_uct = -10000000
        best_child = None

        for c in self.children:
            exploitation = c.q()
            exploration = c.p * param.MCTS_CPUCT * math.sqrt(self.n) / (1 + c.n)

            uct = exploitation + exploration

            if uct > best_uct:
                best_uct = uct
                best_child = c
        
        return best_child

    def backprop(self, value):
        self.n += 1
        self.w += value * self.turn

        if self.parent:
            self.parent.backprop(value)

class Tree:
    def __init__(self):
        self.env = param.SELECTED_ENV()
        self.clear_subtree()

    def clear_subtree(self):
        self.root = Node()
        self.root.turn = self.env.turn
        self.forward_ply = 0
        self.target_node = None

    def select(self, root=None):
        """Returns the next observation to be expanded, or None if
           the target node count is reached."""

        if root is None:
            root = self.root

        if root.n >= param.MCTS_NODES:
            return None

        # Are we at terminal? Backprop again if needed
        tvalue = self.env.terminal()

        if tvalue is not None:
            root.backprop(tvalue)
            self._rewind()
            return self.select()

        # Nonterminal, try and pick a child
        child = root.select()

        if child is None:
            self.target_node = root
            return self.env.observe()

        self.env.push(child.action)
        self.forward_ply += 1

        return self.select(child)
    
    def expand(self, policy, value):
        self.target_node.backprop(value)

        mask = self.env.lmm()

        self.target_node.children = []

        # [ this encourages exploration INITIALLY, but eliminates noise when carrying over trees ]
        #noise = np.zeros((param.PSIZE,))
        #if self.target_node.parent is not None:

        noise = np.random.dirichlet([param.MCTS_NOISE_ALPHA] * param.PSIZE)

        # normalize policy
        policy /= np.linalg.norm(mask * policy)

        for i in range(len(mask)):
            if mask[i] > 0:
                child = Node()

                child.turn = -self.env.turn
                child.action = i
                child.parent = self.target_node
                child.p = policy[i] * (1 - param.MCTS_NOISE_WEIGHT) + noise[i] * param.MCTS_NOISE_WEIGHT

                self.target_node.children.append(child)

        self.target_node = None
        self._rewind()

    def pick(self):
        if self.root.n < param.MCTS_NODES:
            raise RuntimeError('pick() called with bad tree n {}'.format(self.root.n))

        # Select best n
        best_n = 0
        best_action = None

        for c in self.root.children:
            if c.n >= best_n:
                best_n = c.n
                best_action = c.action

        if best_action is None:
            print(str(self.env))
            raise RuntimeError('couldn\'t pick a node, {} {} {} {}'.format([c.n for c in self.root.children], self.env.terminal(), self.env.lmm(), self.root.n))

        self.advance(best_action)
        return best_action

    def advance(self, action):
        selected = None

        if self.root.children is None:
            self.root = Node()
            self.env.push(action)
            return

        if self.target_node is not None:
            raise RuntimeError('advance() called while waiting for expansion')

        for c in self.root.children:
            if c.action == action:
                selected = c
        
        if selected is None:
            print(str(self.env))
            raise RuntimeError('couldn\'t pick a node, {} {} {}'.format([c.n for c in self.root.children], self.env.terminal(), self.root.n))

        self.root = selected
        self.root.parent = None
        self.env.push(action)

    def snapshot(self):
        if self.root.n < param.MCTS_NODES:
            raise RuntimeError('snapshot() called with root n {}'.format(self.root.n))

        out = [0.0] * param.PSIZE
        tn = 0

        for c in self.root.children:
            tn += c.n

        if not self.root.children:
            print(str(self.env))
            raise RuntimeError('snapshot(): no root children')

        for c in self.root.children:
            out[c.action] = c.n / tn

        return out

    def _rewind(self):
        for i in range(self.forward_ply):
            self.env.pop()

        self.forward_ply = 0

    def terminal(self):
        return self.env.terminal()
