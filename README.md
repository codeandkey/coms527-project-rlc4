## Justin Stanley - COMS527 Final Project
A distributed reinforcement learning system targeting Connect-4.

### Starting
This project requires PyTorch 1.2.1+, numpy, and a computer with a GPU available. The learning hyperparameters are in `drl/param.py`.

To start the learning process,
```bash
$ mpirun python drl.py
```

To view the current training progress,
```bash
$ python results.py
```
A window showing the loss over time and model performance will appear, and a file `results.png` will be written with the same information.

### Systems Architecture
![systems](https://raw.githubusercontent.com/codeandkey/coms527-project-rlc4/master/systems.png)

### Model Architecture
![model](https://raw.githubusercontent.com/codeandkey/coms527-project-rlc4/master/network.png)

