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
