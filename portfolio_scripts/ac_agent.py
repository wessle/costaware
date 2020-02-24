import gym
import numpy as np
import torch
from collections import OrderedDict
from time import time, sleep
import os
import sys
import wesutils

import main.utils.utils as utils
from main.core import agents, asset, envs, models


config_path = 'ac_config.yml'

if __name__ == '__main__':

    config = wesutils.load_config(config_path)

    # create variables for all entries in the config file
    for k, v in config.items():
        exec(k + '= v')

    experiment_dir = wesutils.create_logdir(log_dir, algorithm_name,
                                            env_name, config_path)

    # Set the number of threads pytorch can use, seed RNGs
    torch.set_num_threads(num_threads)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    elif len(sys.argv) > 1:
        seed = int(sys.argv[1])
        np.random.seed(seed)
    else:
        seed = np.random.randint(1000)
        np.random.seed(seed)
    print('Seed: {}'.format(seed))

    portfolio = utils.make_portfolio(init_prices,
                                     mean_returns,
                                     stdev_returns,
                                     init_weights,
                                     init_principal)

    env_builder = eval('envs.' + env_name + 'CostAwareEnv')
    if 'Sharpe' in env_name:
        env = env_builder(portfolio)
    else:
        env = env_builder(portfolio, env_target_return)
    env.reset()


    # create agent
    action_dim = len(portfolio)
    state_dim = len(env.state)
    policy = models.DirichletPolicyTwoLayer(state_dim, action_dim,
                                    min_alpha=min_alpha,
                                    max_alpha=max_alpha,
                                    hidden_layer1_size=policy_hidden_units,
                                    hidden_layer2_size=policy_hidden_units)

    v = wesutils.two_layer_net(state_dim, 1, v_hidden_units, v_hidden_units)
    agent = agents.DeepACAgent(buffer_maxlen, batchsize,
                           policy, v,
                           policy_lr, v_lr,
                           init_mu_r, init_mu_c, mu_lr,
                           enable_cuda=enable_cuda,
                           grad_clip_radius=grad_clip_radius)

    if loading_checkpoint:
        agent.load_models(checkpoint_filename)

    end_values = []
    for i in range(1, num_episodes+1):
        average_action = np.zeros(action_dim)
        t0 = time()
        for _ in range(episode_len):
            action = agent.sample_action(env.state)
            average_action += action
            state, reward_cost_tuple, proceed, _  = env.step(action)
            agent.update(reward_cost_tuple, state)
        end_values.append(env.state[0])
        print('Episode {:<6} | ${:>15.2f} | {:.2f}s | {}'.format(
            i, env.state[0],time() - t0,
            (average_action / episode_len).round(decimals=2)))
        env.reset()

        if i % checkpoint_interval == 0:
            agent.save_models(os.path.join(experiment_dir, 'models.pt'))
            np.save(os.path.join(experiment_dir, 'end_values.npy'),
                    np.array(end_values))

# end
