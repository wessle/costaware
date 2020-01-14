import gym
import numpy as np
import torch
from collections import OrderedDict

import main.core.agents as agents
import main.core.asset as asset
import main.core.envs as envs
import main.utils.utils as utils


# asset parameters
price1 = 1
mean1 = 0.05
stdev1 = 0
price2 = 1
mean2 = 0.025
stdev2 = 0
weight1 = 0.5
principal = 1000

portfolio = utils.two_asset_portfolio(price1, mean1, stdev1,
                                      price2, mean2, stdev2,
                                      weight1, principal)
env = envs.SharpeCostAwareEnv(portfolio)
env.reset()


# agent parameters
action_dim = len(portfolio)
state_dim = len(env.state)
buffer_maxlen = 10**6
batchsize = 256
discretization_steps = 10
q_lr = 1e-3
rho_lr = 1e-3

Q = utils.simple_network(state_dim + action_dim, 1)
agent = agents.RVIQLearningBasedAgent(
    buffer_maxlen, batchsize, env.get_actions(discretization_steps),
    Q, q_lr, rho_lr)


# training session parameters
num_episodes = 100
episode_len = 10

for i in range(num_episodes):
    for _ in range(episode_len):
        action = agent.sample_action(env.state)
        state, reward_cost_tuple, proceed, _  = env.step(action)
        agent.update(reward_cost_tuple, state)
    print(f'Episode{i} ending capital: {env.state[0]}') 
    env.reset()




# end
