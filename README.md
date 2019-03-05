# Differentiable Decision Trees

## Installation

### Required System Dependencies

This code has been run and developed solely on Ubuntu 18.04. We recommend 
using the same operating system in order to replicate our results.

You will need the following system dependencies to run our code:

- [Conda](https://www.anaconda.com/)
- [GNU Parallel](https://www.gnu.org/software/parallel/)
- [Bash](https://www.gnu.org/software/bash/)

### Installing Python Dependencies
Run the following commands to install the environment in the 
requirements.yaml and activate it.
```bash
conda env create -f requirements.yaml
conda activate DDT
```

### Running Tests

Make the tests directory your working directory and execute run_tests.sh

```bash
cd ./tests
./run_tests.sh
```

If the tests run the environment should have been installed correctly and you
 can start running the experiments.

## Experiments

### Running Experiments 

There are scripts to run experiments in parallel located in the exp_cmds 
directory. All the scripts inside that directory assume exp_cmds is the 
current directory.

1. Supervised Learning Experiments:
Run the following command:
```bash
cd ./exp_cmds
./supervised_learning_command.sh
```

As your output you should see A cancer, ttt and cesarean directories 
generated under the plots directory. Each of these corresponds to one of the 
UCI datasets used for comparison. For each UCI dataset you should see a tree 
and ddt directory showing the results for a regular decision tree and a 
differentiable decision tree respectively.

2. Reinforcement Learning Experiments:

The following commands will run the experiments used to collect the MLP 
baselines as well as the differentiable decision tree experiments used in the
 paper. Notice that this command may take a long time to execute.

```bash
cd ./exp_cmds
./collect_baselines.sh
./command_runner.sh
```

The output of these commands should be a series of folders in the 
tensorboards folder corresponding to each experiment. To view the results run
 the following command from the toplevel directory:
 
 ```bash
 
 tensorboard --logdir tensorboards
 
 ```
 
 This should allow you to view the training curves and relevant data for each
  of the experiments.

### Viewing Results

To replicate the figures in the paper please run the following command:

```bash
cd exp_cmds
./exp_plotter
```

Notice that for this command to you must arrange the tensorboard results for 
each experiment such that your directory structure looks like this:
tensorboards/final_exp/$$EXPERIMENT_NAME$$/$$MODEL$$/$$SEED_FOLDER$$.
As provided the exp_plotter script expects the follwing experiment names:
LunarLander, CartPole and Acrobot.
A sample directory structure could look like this:
```bash
tensorboards/final_exp/Acrobot/ddt4/env.Acrobot-v1-m.ddt.4-reg.b0.0001.m1.01859.tleaf.mdnever-ceps.1.0-alpha.b1.0.m1.0.inc0.0-lr.0.03-alg.ppo-init.uniform-s.1
```
## Code Structure

### PPO/ACKTR/A2C Implementation

Please note that we use lightly modified version of [ikostrikov's 
implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) of
 PPO, ACKTR and A2C. 

### Pipelines

1. pg_pipeline.py

This file contains the reinforcement learning pipeline we use.

2. sl_pipeline.py

This file contains the supervised learning pipeline.


### Models

We provide a Linear Policy, an MLP and two implementations of the 
differentiable decision trees. DDTN uses a recursive implementation which is 
quite slow. For all of our experiments we use FDDTN which takes advantage of 
a full tree structure to parallelize tree execution in the GPU. The 
ActorCritic model is simply a wrapper to be able to use ikostrikov's 
PPO/ACKTR/A2C implementation.

### Data and Visualization

1. The data folder contains the UCI cesarean datset.
2. Plot.py generates the learning curves with error shades shown in the paper. 



 
