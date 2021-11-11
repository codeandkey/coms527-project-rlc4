# Trainer entry point.

from . import cluster
from . import model
from . import param
from . import util

def start():
    util.log('Starting trainer task.')

    if not model.load():
        model.generate()
        model.save()

    for dest in cluster.inferencers:
        cluster.comm.send((cluster.comm.rank, param.MSG_RELOAD), dest)