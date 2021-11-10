# General utilities

import datetime
import identity

def log(x):
    dtime = datetime.datetime.now()
    
    print('{} {} [{}]: {}'.format(
        dtime,
        identity.rank,
        identity.task,
        x
    ))