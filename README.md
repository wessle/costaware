# costaware

A repository for performing optimization and reinforcement learning for Cost-Aware Markov Decision Processes (CAMDPs). This is the supplementary repo for the article, "Reinforcement Learning for Cost-Aware Markov Decision Processes", to be published in the _Proceedings of the 38th International Conference on Machine Learning_ in 2021.

## Details

Ratio maximization has applications in areas as diverse as finance, reward shaping for reinforcement
learning (RL), and the development of safe artificial intelligence.

This repository implements new ratio-maximization RL algorithms as described in the paper. It also implements an easy-to-use suite for setting up experiments on CAMDPs.

### Contributions

* Cost-aware relative value iteration Q-learning (CARVI Q-learning), an adaptation of traditional RVI Q-learning for CAMDPs. This repository implements CARVI Q-learning for an agent with tabular architecture and for agents with neural-network architectures. 
* Cost-aware actor-critic (CAAC), an adaptation of the traditional actor-critic learning for CAMDPs. This repository also implements CAAC for agents with linear approximators under a variety of feature architectures.

### Features

* "Batteries included" agents fully-implemented for CARVI Q-learning and CAAC, as well as a suite of pre-defined policy, value, and Q-function architectures.
* Built-in CAMDP environments that adhere to the `gym.Env` API.
* Cost-aware variations of classic control problems, such as `MountainCar`, `Acrobat`, `Pendulum`, and `CartPole`.
* Configurable suite for performing cost-aware RL experiments, with customizable configuration files and  simple plotting.

### Requirements

This repository uses Python 3. Package details can be found in the `requirements.txt` file. Major third-party libraries used in this repository include  

* [PyTorch](https://pytorch.org/)
* [Ray](https://ray.io/)
* PyData suite: [NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

## Installation

#### Virtual environment setup

Set up your preferred environment and activate it, e.g. do
```bash
python -m venv .venv
source .venv/bin/activate
```
After cloning, simply `cd` into the repo and do `pip install -e .`.

#### Container setup

Using `docker` or `podman`, first build the image with

```bash
docker build -t=costaware -f Dockerfile
```

then run it with your local `costaware` repo mounted in the container using

```bash
docker run -it --rm --privileged -v COSTAWARE_PATH:/home/costaware costaware
```

Any changes made to `/home/costaware` in the container will be reflected
on the host machine in `COSTAWARE_PATH` and vice versa.

## Getting Started

The main directory contains a script called `demo.py` that can be used to run the same experiments described in the experimental portion of the paper. Based on the argument passed to it, `demo.py` uses the scripts and configuration files in the `examples` directory to launch an experiment. For example, if

```python demo.py synthetic```

is called, the experiment script `synthetic_runner.py` --- which trains tabular Cost-Aware RVI Q-learning (CARVI) Q-learning and Cost-Aware Actor-Critic (CAAC) with linear function approximation on a synthetic CAMDP environment --- will be run using the experiment specified by the file `synthetic_config.yml`. Similarly, if

```python demo.py deep_Q```

is called, `deep_Q_runner.py` --- which trains a neural network version of CARVI Q-learning on a cost-aware version of a classic Gym environment --- will be run using the experiment specified in `deep_Q_config.yml`. When the experiment is finished running, `demo.py` will also generate a plot of the experiment performance.

### Configuration Files

The configuration files `synthetic_config.yml` and `deep_Q_config.yml` contain all the experiment parameters and hyperparameters that one needs to set up and run various types of experiments. In a nutshell, a configuration file specifies an **experiment**, which is composed of one or more **trials**.

A **trial** consists of
* an environment to be trained on,
* the type of agent to be trained,
* algorithm hyperparameters,
* the directory to save output to.

An **experiment** consists of
* one or more trial to be run,
* the number of replications of each trial to be run,
* a list of the available resources (CPUs and GPUs),
* the resources to be allocated to each trial replication.

### Running Experiments

A typical experiment launched by `demo.py` will consist of several replications of the same trial. The output of these replications can then be used to generate nice plots showing average performance across all trials. After verifying that your machine has sufficient computational resources to match what the configuration file asks for, the trials that make up the experiment are created and run in parallel using Ray. This makes it easy run large numbers of trials at once if you have sufficient resources.

After you've run `demo.py` a few times and see how things work, feel free to start playing with the configuration files to recreate the experiments from the paper or make your own.
