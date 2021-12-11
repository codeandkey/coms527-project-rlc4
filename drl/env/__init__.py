# Defines environment type

import uuid

class Environment:
    """Generic environment superclass."""
    def __init__(self):
        self.tag = uuid.uuid4()

    def push(self, action):
        """Performs a new action."""
        raise RuntimeError('push() method missing!')

    def pop(self):
        """Unperforms the last action taken."""
        raise RuntimeError('pop() method missing!')

    def lmm(self):
        """Returns a 1D numpy array p with values in (0, 1),
           such that p[a] == 1 if a is a valid action."""
        raise RuntimeError('lmm() method missing!')

    def observe(self):
        """Returns an n-dimensional numpy array describing the state of the
           environment."""
        raise RuntimeError('observe() method missing!')

    def terminal(self):
        """Returns a terminal value if the environment is in a terminal state,
           otherwise returns None."""
        raise RuntimeError('terminal() method missing!')
