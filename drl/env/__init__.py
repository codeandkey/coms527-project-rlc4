# Defines environment type

import uuid

class Environment:
    def __init__(self):
        self.tag = uuid.uuid4()

    def push(self, action):
        raise RuntimeError('push() method missing!')

    def pop(self):
        raise RuntimeError('pop() method missing!')

    def lmm(self):
        raise RuntimeError('lmm() method missing!')

    def observe(self):
        raise RuntimeError('observe() method missing!')

    def terminal(self):
        raise RuntimeError('terminal() method missing!')
