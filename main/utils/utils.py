"""Provide some utilities, such as a replay buffer for RL agents."""

import numpy as np
import torch
import random
from collections import deque
import yaml

from gym import spaces


def load_config(filename):
    """Load and return a config file."""

    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

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
        
        
class AllocationSimplex(spaces.Box):
    """
    Action space of possible portfolio allocations.
    
    Allocations should sum to 1, but we allow those getting close.
    """

    def __init__(self, num_assets):
        super.__init__(0.0, 1.0, shape=(num_assets,))

    def sample(self):
        """Sample uniformly from the unit simplex."""

        v = sorted(np.random.uniform(size=self.shape[0]-1)).append(1.0)
        allocations = [v[i] if i == 0 else v[i] - v[i-1] for i in range(len(v))]
        return np.array(allocations)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return x.shape == self.shape and np.all(x >= self.low) \
                and np.all(x <= self.high) and np.isclose(np.sum(x), 1.0)

    def __repr__(self):
        return "AllocationSimplex" + str(self.shape[0])

    def __eq__(self, other):
        return isinstance(other, AllocationSimplex) and self.shape == other.shape
    



# end
