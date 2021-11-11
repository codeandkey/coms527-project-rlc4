# General utilities

from . import cluster

import datetime

def log(x):
    dtime = datetime.datetime.now()
    
    print('{} {} [{}]: {}'.format(
        dtime,
        cluster.rank,
        cluster.task,
        x
    ))