from . import cluster

def start():
    # Resolve identity immediately
    cluster.resolve()

    # Print our identity
    print('Node {} : {}'.format(cluster.rank, cluster.task))

    # Load entry point
    if cluster.task == 'trainer':
        print('Starting trainer.')
        from . import trainer
        trainer.start()
    elif cluster.task == 'inference':
        print('Starting inference.')
        from . import inference
        inference.start()