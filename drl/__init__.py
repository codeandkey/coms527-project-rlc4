from . import cluster

def start():
    """Starts the training process."""
    # Resolve identity immediately
    cluster.resolve()

    # Print our identity
    #print('Node {} : {}'.format(cluster.rank, cluster.task))

    # Load entry point
    if cluster.task == 'train':
        #print('Starting trainer.')
        from . import trainer
        trainer.start()
    elif cluster.task == 'infer':
        #print('Starting inference.')
        from . import inference
        inference.start()
    elif cluster.task == 'actor':
        #print('Starting actor.')
        from . import actor
        actor.start()
