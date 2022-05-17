from . import Environment

import numpy as np

class Connect4(Environment):
    width = 7
    height = 6
    features = 3
    psize = 7

    def __init__(self):
        """Initializes a new connect-4 environment. 'X' turn corresponds to
           1, 'O' to -1. 'X' to move first."""
        super()
        
        self.cells = [0] * 42
        self.turn = 1
        self.actions = []

    def cell(self, x, y):
        """(internal) Helper method to query the state of a game cell."""
        return self.cells[x + y * 7]

    def lmm(self):
        """Returns a mask of legal moves."""
        out = np.zeros(7)

        for c in range(7):
            out[c] = 1 if self.cells[35 + c] == 0 else 0

        return out

    def __str__(self):
        """Converts this environment to a printable string."""

        rows = []
        cmap = {0: 'O', 1: '.', 2: 'X'}

        for y in reversed(range(6)):
            rows.append(''.join([str(cmap[self.cell(x, y) + 1]) for x in range(7)]))

        out =  '+-------+\n'
        out += '|{}| {} to move\n'.format(rows[0], 'X' if self.turn > 0 else 'O')
        out += '|{}| {}\n'.format(rows[1], ' '.join(map(str, reversed(self.actions[-10:]))) + ' ...')
        out += '|' + '|\n|'.join(rows[2:]) + '|\n'
        out +=  '+-------+'

        return out

    def push(self, action):
        """Performs an action. <action> must be a legal move."""
        for y in range(6):
            ind = y * 7 + action

            if self.cells[ind] == 0:
                self.cells[ind] = self.turn
                self.actions.append(action)
                self.turn = -self.turn
                return

        raise RuntimeError('illegal move')

    def pop(self):
        """Unperforms the last action."""
        last = self.actions.pop()
        self.turn = -self.turn

        for r in reversed(range(6)):
            ind = r * 7 + last

            if self.cells[ind] != 0:
                self.cells[ind] = 0
                return


    def observe(self):
        """Returns a 7x6x3 numpy array describing the state of the game."""
        out = np.zeros((7, 6, 3))

        for x in range(7):
            for y in range(6):
                if self.turn == -1:
                    out[x, y, 2] = 1

                cell = self.cell(x, y)

                if cell == 1:
                    out[x, y, 0] = 1
                elif cell == -1:
                    out[x, y, 1] = 1

        return out

    def terminal(self):
        """Returns 1 if 'X' wins, -1 if 'O' wins, 0 if draw, or None if the
           game is ongoing."""

        # Check for 4-in-a-row
        def check(segment):
            values = [self.cells[x] for x in segment]

            for x in values[1:]:
                if x != values[0] or x == 0:
                    return None

            return values[0]

        def segments():
            # Horizontal segments
            for x in range(4):
                for y in range(6):
                    root = x + y * 7
                    yield range(root, root + 4)

            # Vertical segments
            for x in range(7):
                for y in range(3):
                    root = x + y * 7
                    yield range(root, root + 28, 7)

            # Northeast segments
            for x in range(4):
                for y in range(3):
                    root = x + y * 7
                    yield range(root, root + 32, 8)

            # Northwest segments
            for x in range(3, 7):
                for y in range(3):
                    root = x + y * 7
                    yield range(root, root + 24, 6)

        # Test segments for terminal
        for s in segments():
            result = check(s)

            if result is not None: return result

        # Check if draw
        for c in self.cells:
            if c == 0: return None
        
        return 0
