import gym
import numpy as np
import torch
from collections import OrderedDict
from time import time

import main.core.agents as agents
import main.core.asset as asset
import main.core.envs as envs
import main.utils.utils as utils


# asset parameters
price1 = 1
mean1 = 0.05
stdev1 = 0
price2 = 1
mean2 = 0.0
stdev2 = 0
weight1 = 0.5
principal = 100

portfolio = utils.two_asset_portfolio(price1, mean1, stdev1,
                                      price2, mean2, stdev2,
                                      weight1, principal)
env = envs.SharpeCostAwareEnv(portfolio)
# env = envs.OmegaCostAwareEnv(portfolio, np.mean([mean1, mean2]))
# env = envs.SortinoCostAwareEnv(portfolio, np.mean([mean1, mean2]))
env.reset()


# agent parameters
action_dim = len(portfolio)
state_dim = len(env.state)
buffer_maxlen = 10**6
batchsize = 256
discretization_steps = 4
q_lr = 1e-3
rho_lr = 1e-3
eps = 0.3
checkpoint_filename = '../data/qlearning_checkpoint.pt'
loading_checkpoint = False

Q = utils.simple_network(state_dim + action_dim, 1)
agent = agents.RVIQLearningBasedAgent(
    buffer_maxlen, batchsize, env.get_actions(discretization_steps),
    Q, q_lr, rho_lr, eps=eps)
if loading_checkpoint:
    agent.load_models(checkpoint_filename)


# training session parameters
num_episodes = 100
episode_len = 100
checkpoint_interval = 10

average_action = 0.5*np.ones(2)
for i in range(num_episodes):
    t0 = time()
    for _ in range(episode_len):
        action = agent.sample_action(env.state)
        average_action = np.mean([average_action, action], axis=0)
        state, reward_cost_tuple, proceed, _  = env.step(action)
        agent.update(reward_cost_tuple, state)
    print('Episode {}: ${:.2f}, {:.2f}s, {}'.format(i,
                                                    env.state[0],
                                                    time() - t0, 
                                                    average_action))
    env.reset()

    if i % checkpoint_interval == 0:
        agent.save_models(checkpoint_filename)




# end
