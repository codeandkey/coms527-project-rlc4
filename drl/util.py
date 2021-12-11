
# General utilities

from . import cluster

import datetime

def log(x):
    """Writes a log to stdout with datetime and node information included."""
    dtime = datetime.datetime.now()
    
    print('{} {} [{}]: {}'.format(
        dtime,
        cluster.rank,
        cluster.task,
        x
    ))