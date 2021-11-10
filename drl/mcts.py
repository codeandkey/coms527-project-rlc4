# MCTS tree/node types

import param

import numpy as np
import random

class Node:
    def __init__(self):
        self.n = 0
        self.w = 0

    def select(self):
        if not self.children:
            return None

    def backprop(self, value):
        self.n += 1
        self.w += value

        if self.parent:
            self.parent.backprop(-value)

class Tree:
    def __init__(self):
        self.root = Node()
        self.env = param.SELECTED_ENV()

    def select(self, root=None):
        """Returns the next observation to be expanded, or None if
           the target node count is reached."""

        if root is None:
            root = self.root

        if root.n >= param.TARGET_NODES:
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
            return self.env.observe()

        self.env.push(child.action)
        self.forward_ply += 1

        return self.select(child)
    
    def expand(self, policy, value):
        self.target_node.backprop(value)

        mask = self.env.lmm()

        self.target_node.children = []

        for i in range(len(mask)):
            if mask[i] > 0:
                child = Node()

                child.action = i
                child.parent = self.target_node
                child.p = policy[i]

                self.target_node.children.append(child)

        self.target_node = None
        self._rewind()

    def pick(self):
        # Select best n
        best_n = 0
        best_action = 0
        best_c = None

        for c in self.root.children:
            if c.n > best_n:
                best_n = c.n
                best_action = c.action
                best_c = c

        self.root = best_c
        self.root.parent = None
        self.env.push(best_action)

        return best_action

    def _rewind(self):
        for i in range(self.forward_ply):
            self.env.pop()

        self.forward_ply = 0

    def terminal(self):
        return self.env.terminal()
