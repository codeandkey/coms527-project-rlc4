# Methods for managing identity

import torch
from mpi4py import MPI
import numpy as np

task = None

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def resolve():
    """Resolves identity of each node. Should be called before any other
       communications methods in each node."""

    cuda_stat = np.empty(comm.Get_size(), dtype='i')

    has_cuda = np.empty(1, dtype='i')
    has_cuda[0] = 1 if torch.cuda.is_available() else 0

    comm.Allgather([has_cuda, MPI.INT], [cuda_stat, MPI.INT])

    print('CUDA status: {}'.format(cuda_stat))

    # TEMP: just pick 1 training, 1 inference, and N-2 actors
    if len(cuda_stat) < 3:
        raise RuntimeError('At least 3 nodes required')

    global task
    if rank == 0:
        task = 'training'
    elif rank == 1:
        task = 'inference'
    else:
        task = 'actor'
