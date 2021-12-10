from . import Environment

import numpy as np

class Connect4(Environment):
    width = 7
    height = 6
    features = 3
    psize = 7

    def __init__(self):
        super()
        
        self.cells = [None] * 42
        self.turn = 1
        self.actions = []

    def cell(self, x, y):
        return self.cells[x + y * 7]

    def lmm(self):
        out = np.zeros(7)

        for c in range(7):
            out[c] = 1 if self.cells[35 + c] is None else 0

        return out

    def __str__(self):
        out = ''

        for y in reversed(range(6)):
            for x in range(7):
                if self.cell(x, y) == 1:
                    out += 'X'
                elif self.cell(x, y) == -1:
                    out += 'O'
                else:
                    out += '.'
            out += '\n'

        return out + 'hist: {}'.format(self.actions)

    def push(self, action):
        for r in range(6):
            ind = r * 7 + action

            if self.cells[ind] is None:
                self.cells[ind] = self.turn
                self.actions.append(action)
                self.turn = -self.turn
                return

        raise RuntimeError('illegal move')

    def pop(self):
        last = self.actions.pop()
        self.turn = -self.turn

        for r in reversed(range(6)):
            ind = r * 7 + last

            if self.cells[ind] is not None:
                self.cells[ind] = None
                return


    def observe(self):
        out = np.zeros((7, 6, 3))

        for x in range(7):
            for y in range(6):
                if self.turn == -1:
                    out[x, y, 2] = 1

                cell = self.cells[x + 7 * y]

                if cell is None:
                    continue

                if cell == 1:
                    out[x, y, 0] = 1
                elif cell == -1:
                    out[x, y, 1] = 1

        return out

    def terminal(self):
        # Check for 4-in-a-row
        def check(segment):
            values = [self.cells[x] for x in segment]
            first = values[0]

            if first is not None:
                for x in values[1:]:
                    if x != first:
                        return None

            return first

        def segments():
            # Horizontal segments
            for x in range(4):
                for y in range(6):
                    root = x + y * 7
                    yield list(range(root, root + 4))

            # Vertical segments
            for x in range(7):
                for y in range(3):
                    root = x + y * 7
                    yield list(range(root, root + 28, 7))

            # Northeast segments
            for x in range(4):
                for y in range(3):
                    root = x + y * 7
                    yield list(range(root, root + 32, 8))

            # Northwest segments
            for x in range(3, 7):
                for y in range(3):
                    root = x + y * 7
                    yield list(range(root, root + 24, 6))

        # Test segments for terminal
        for s in segments():
            result = check(s)

            if result is not None:
                return result

        # Check if draw
        if None in self.cells:
            return None
        
        return 0
