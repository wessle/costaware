import gym
import numpy as np
import torch
from collections import OrderedDict
from time import time, sleep

import main.core.agents as agents
import main.core.asset as asset
import main.core.envs as envs
import main.core.models as models
import main.utils.utils as utils


if __name__ == '__main__':

    # asset parameters
    price1 = 1
    mean1 = 0.0003
    stdev1 = 0
    price2 = 1
    mean2 = 0.000
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
    min_alpha = 0.01
    max_alpha = 50
    buffer_maxlen = 10**6
    batchsize = 256
    policy_lr = 1e-2
    v_lr = 1e-2
    enable_cuda = True
    grad_clip_radius = None # set to None for no clipping
    checkpoint_filename = '../data/ac_checkpoint.pt'
    loading_checkpoint = False
    policy_hidden_units = 256
    v_hidden_units = 256

#    policy = models.DirichletPolicySingleLayer(state_dim, action_dim,
#                                    min_alpha=min_alpha,
#                                    max_alpha=max_alpha,
#                                    hidden_layer_size=policy_hidden_units)

    policy = models.DirichletPolicyTwoLayer(state_dim, action_dim,
                                    min_alpha=min_alpha,
                                    max_alpha=max_alpha,
                                    hidden_layer1_size=policy_hidden_units,
                                    hidden_layer2_size=policy_hidden_units)

    v = utils.two_layer_net(state_dim, 1, v_hidden_units, v_hidden_units)
    agent = agents.ACAgent(buffer_maxlen, batchsize,
                           policy, v,
                           policy_lr, v_lr,
                           enable_cuda=enable_cuda,
                           grad_clip_radius=grad_clip_radius)

    if loading_checkpoint:
        agent.load_models(checkpoint_filename)


    # training session parameters
    num_episodes = 1000
    episode_len = 365
    checkpoint_interval = 100

    for i in range(num_episodes):
        t0 = time()
        average_action = 0.5*np.ones(2)
        for _ in range(episode_len):
            action = agent.sample_action(env.state)
            average_action = np.mean([average_action, action], axis=0)
            state, reward_cost_tuple, proceed, _  = env.step(action)
            agent.update(reward_cost_tuple, state)
        print('Episode {:<6} | ${:>15.2f} | {:.2f}s | {}'.format(i,
                                                        env.state[0],
                                                        time() - t0,
                                                        average_action))
        print(agent.pi.forward(torch.FloatTensor(env.state).to('cuda')).cpu().detach().numpy())
        env.reset()

        if i % checkpoint_interval == 0:
            agent.save_models(checkpoint_filename)



# end
