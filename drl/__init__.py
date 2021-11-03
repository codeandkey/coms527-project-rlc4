from . import identity

def start():
    # Resolve identity immediately
    identity.resolve()

    # Print our identity
    print('Node {} : {}'.format(identity.rank, identity.task))
