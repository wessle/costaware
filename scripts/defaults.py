import numpy as np
import main.core.agents as agents
from collections.abc import Mapping
from copy import deepcopy


# Functions for use in defining synthetic cost-aware MDPs

goal_state = 0


def r1(s, a):
    return s**3

def r2(s, a):
    return s + a

def r3(s, a):
    return (s % 2) * (a % 2)

def r4(s, a):
    return 100 * (s == goal_state)

def r5(s, a):
    return 1 * (s % 2 == 0)


def c1(s, a):
    return max(1, s * a)

def c2(s, a):
    return 1 / max(1, s*a)

def c3(s, a):
    return 1 + (a % 3 - 1)**2

def c4(s, a):
    return np.exp(-s)

def c5(s, a):
    return 1

def c6(s, a):
    return 1 * (s == goal_state) + 10 * (s != goal_state)

functions = {
    'rewards': {'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4, 'r5': r5},
    'costs':   {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'c6': c6}
}


# default configuration

default_configs = {
    'num_states':       [5, 10, 20],
    'num_actions':      [5, 10, 20],
    'rewards_fn':       'r4',
    'costs_fn':         'c6',
    'transition_seed':  1994,  # null for random seed
    'training_seed':    None,  # null for random seed

    # Actor/Critic defaults
    'algorithm_config': {
        'LinearAC': {
            'class': agents.LinearACAgent,
            'kwargs': {
                'policy_lr':        0.0001,
                'v_lr':             0.001,
                'init_mu_r':        1,
                'init_mu_c':        1,
                'mu_lr':            0.0001,
                'mu_floor':         0.01,
                'grad_clip_radius': None,  # null for no clipping
            }

        },
        'TabularQ': {
            'class': agents.TabularQAgent,
            'kwargs': {
                'q_lr':     0.001,
                'rho_lr':   0.00001,
                'rho_init': 0,
                'eps':      0.2,
            }
        }
    },

    # experiment
    'env_name':               'MDP',
    'num_steps':              30000,# 500000,
    'print_interval':         10000,
    'logging':                True,
    'mc_testing':             False,
    'mc_testing_time_window': 50000,
    'moving_average_width':   1000,
}

