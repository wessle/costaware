import numpy as np
import torch
import random
from collections import deque
import yaml
import pickle
import os
from datetime import datetime
from shutil import copyfile
from typing import List, Tuple
from gym import spaces
import matplotlib.pyplot as plt


########## Less important functions for use in testing ##########

def single_layer_net(input_dim, output_dim, hidden_layer_size=256):
    """
    Generate a fully-connected single-layer network for quick use.
    """
    net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer_size, output_dim))
    return net

def two_layer_net(input_dim, output_dim,
                  hidden_layer1_size=256,
                  hidden_layer2_size=256,
                  activation='ReLU'):
    """
    Generate a fully-connected two-layer network for quick use.
    """
    
    activ = eval('torch.nn.' + activation)

    net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_layer1_size),
        activ(),
        torch.nn.Linear(hidden_layer1_size, hidden_layer2_size),
        activ(),
        torch.nn.Linear(hidden_layer2_size, output_dim)
    )
    return net

import main.core.portfolio as portfolio
import main.core.asset as asset

def make_portfolio(init_prices,
                   mean_returns,
                   stdev_returns,
                   init_weights,
                   init_principal,
                   asset_class='SimpleAsset'):
    """
    Take lists of asset parameters, then return a corresponding portfolio.
    """

    asset_params = zip(init_prices, mean_returns, stdev_returns)
    assets = [eval('asset.' + asset_class)(*param) for param in asset_params]
    return portfolio.Portfolio(assets, init_weights, init_principal)


def plot_average_fig(pickle_name, plot_filename):
    """
    Read in pickle files (all with the same name) from all
    subdirectories in the current directory containing lists of returns
    and plot their average.
    """

    for _, _, files in os.walk('.'):
        returns = [load_object(file) for file in file if file == pickle_name]

    


########## Important functions ##########

def load_config(filename):
    """Load and return a config file."""

    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_logdir(directory, algorithm, env_name, config_path):
    """
    Create a directory inside the specified directory for
    logging experiment results and return its path name.

    Include the environment name (Sharpe, etc.) in the directory name,
    and also copy the config
    """

    experiment_dir = f"{directory}/{algorithm}-{env_name}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if config_path is not None:
        copyfile(config_path, f"{experiment_dir}/config.yml")

    return experiment_dir

def save_object(obj, filename):
    """Save an object, e.g. a list of returns for an experiment."""
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(filename):
    """Load an object, e.g. a list of returns to be plotted."""
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def array_to_tensor(array, device):
    """Convert numpy array to tensor."""

    return torch.FloatTensor(array).to(device)

def arrays_to_tensors(arrays, device):
    """Convert iterable of numpy arrays to tuple of tensors."""
    # TODO: use *args in place of arrays
    
    return tuple([array_to_tensor(array, device) for array in arrays])

def copy_parameters(model1, model2):
    """Overwrite model1's parameters with model2's."""
    
    model1.load_state_dict(model2.state_dict())

def generate_simplex(dimension: int, steps: int) -> 'np.ndarray':
    """
    Generates an array of uniformly spaced grid points on the unit simplex in a
    given dimension and under a specified level of coarseness.

    The generation is completely deterministic; given a fixed input
    specification, the function always returns the same result.

    This solution always selects the simplex vertices of the form

        v_i = (0, ..., 0, i, 0, ..., 0)  # 1 in the ith coordinate

    and computes the spacing so that the remaining points lie on a uniform
    lattice between the vertices.

    In full generality, the solution is necessarily primitive recursive.

    Params
    ------
    dimension:    float
                  The number of coordinates for each point. In other words, the
                  dimension of the space in which the simplex inhabits. E.g.,
                  the unit simplex in R^3, the triangle, corresponds to
                  dimension=3.
    steps:        int
                  The number of points to be spaced evenly between vertices of
                  the unit simplex. This works like numpy.linspace.

    Return
    ------
    simplex_grid: numpy.ndarray
                  A numpy array of grid points spaced uniformly according to the
                  specifications above.
    """
    def kernel(num: int, terms: int) -> List[Tuple]:
        """
        Recursive kernel function. This computes the set of "compositions" of
        the number `num` with exactly `terms-1` terms.

        For example, the ways to write the number 4 as a sum of 3 nonnegative
        integers are

            0 + 0 + 4 | 1 + 0 + 3 | 2 + 0 + 2 | 3 + 0 + 1 | 4 + 0 + 0 |
            0 + 1 + 3 | 1 + 1 + 2 | 2 + 1 + 1 | 3 + 1 + 0 |
            0 + 2 + 2 | 1 + 2 + 1 | 2 + 2 + 0 |
            0 + 3 + 2 | 1 + 3 + 0 | 
            0 + 4 + 0 |

        This can be generated by running kernel(3, 4).

        Note: The off by one issue is confusing to me but I am trying to avoid
        thinking through the mathematics too carefully.

        Params
        ------
        num:          int
                      The number to be broken down.
        terms:        int
                      The number of integers to break `num` into.

        Returns
        -------
        compositions: list
                      A list of compositions of `num` composed of `terms-1`
                      terms.

        """
        if terms == 0:
            return [(num,)]
        else:
            return [(i, *j) for i in range(num+1) for j in kernel(num-i, terms-1)]


    return np.array(kernel(steps, dimension-1)) / steps


class Buffer:
    """
    Circular experience replay buffer for cost-aware environments.
    
    Note that reward-cost pairs are stored as a tuple.

    The current sampling procedure is inefficient and should be improved.
    """
    
    def __init__(self, max_length):
        self.buffer = deque(maxlen=max_length)
        
    def sample_batch(self, batch_size):
        # TODO: make this more efficient
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, costs, next_states = [], [], [], [], []
        for elem in batch:
            s, a, (r, c), s2 = elem
            states.append(s)
            actions.append(a)
            rewards.append(r)
            costs.append(c)
            next_states.append(s2)
        return (states, actions, rewards, costs, next_states)
    
    def add(self, sample):
        self.buffer.append(sample)

    def __len__(self):
        return len(self.buffer)
        
        



# end
