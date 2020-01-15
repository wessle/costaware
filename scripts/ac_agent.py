import gym
import numpy as np
import torch
from collections import OrderedDict
from time import time

import main.core.agents as agents
import main.core.asset as asset
import main.core.envs as envs
import main.core.models as models
import main.utils.utils as utils


# asset parameters
price1 = 1
mean1 = 0.0
stdev1 = 0
price2 = 1
mean2 = 0.05
stdev2 = 0
weight1 = 0.5
principal = 100

portfolio = utils.two_asset_portfolio(price1, mean1, stdev1,
                                      price2, mean2, stdev2,
                                      weight1, principal)
env = envs.SharpeCostAwareEnv(portfolio)
# env = envs.SortinoCostAwareEnv(portfolio, np.mean([mean1, mean2]))
env.reset()


# agent parameters
action_dim = len(portfolio)
state_dim = len(env.state)
min_alpha = 0.001
max_alpha = 100
buffer_maxlen = 10**6
batchsize = 256
policy_lr = 1e-3
v_lr = 1e-3
checkpoint_filename = '../data/ac_checkpoint.pt'
loading_checkpoint = True

policy = models.DirichletPolicy(state_dim, action_dim,
                                min_alpha=min_alpha,
                                max_alpha=max_alpha)
v = utils.simple_network(state_dim, 1)
agent = agents.ACAgent(buffer_maxlen, batchsize,
                       policy, v,
                       policy_lr, v_lr)
if loading_checkpoint:
    agent.load_models(checkpoint_filename)


# training session parameters
num_episodes = 50
episode_len = 100
checkpoint_interval = 10

for i in range(num_episodes):
    t0 = time()
    for _ in range(episode_len):
        action = agent.sample_action(env.state)
        state, reward_cost_tuple, proceed, _  = env.step(action)
        agent.update(reward_cost_tuple, state)
    print('Episode {}: ${:.2f}, {:.2f}s'.format(i,
                                                env.state[0],
                                                time() - t0)) 
    print(agent.sample_action(env.state))
    env.reset()

    if i % checkpoint_interval == 0:
        agent.save_models(checkpoint_filename)



# end
