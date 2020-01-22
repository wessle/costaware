import gym
import numpy as np
import torch
from collections import OrderedDict
from time import time
import os

import main.core.agents as agents
import main.core.asset as asset
import main.core.envs as envs
import main.utils.utils as utils


if __name__ == '__main__':

    config_path = 'q_config.yml'

    config = utils.load_config(config_path)

    # experiment parameters
    algorithm_name = config['algorithm_name']
    env_name = config['env_name']
    checkpoint_interval = config['checkpoint_interval']
    log_dir = config['log_dir']
    num_episodes = config['num_episodes']
    episode_len = config['episode_len']
    experiment_dir = utils.create_logdir(log_dir, algorithm_name,
                                         env_name, config_path)
    loading_checkpoint = config['loading_checkpoint']

    # asset parameters
    price1 = config['price1']
    mean1 = config['mean1']
    stdev1 = config['stdev1']
    price2 = config['price2']
    mean2 = config['mean2']
    stdev2 = config['stdev2']
    weight1 = config['weight1']
    principal = config['principal']
    env_name = config['env_name']

    portfolio = utils.two_asset_portfolio(price1, mean1, stdev1,
                                          price2, mean2, stdev2,
                                          weight1, principal)

    env_builder = eval('envs.' + env_name + 'CostAwareEnv')
    if env_name == 'Sharpe':
        env = env_builder(portfolio)
    else:
        env = env_builder(portfolio, np.mean([mean1, mean2]))
    env.reset()


    # agent parameters
    action_dim = len(portfolio)
    state_dim = len(env.state)
    buffer_maxlen = config['buffer_maxlen']
    batchsize = config['batchsize']
    discretization_steps = config['discretization_steps']
    q_lr = config['q_lr']
    rho_lr = config['rho_lr']
    eps = config['eps']
    enable_cuda = config['enable_cuda']
    grad_clip_radius = config['grad_clip_radius']
    rho_clip_radius = config['rho_clip_radius']
    checkpoint_filename = config['checkpoint_filename']
    loading_checkpoint = config['loading_checkpoint']
    Q_hidden_units = config['Q_hidden_units']

    Q = utils.two_layer_net(state_dim + action_dim, 1,
                            Q_hidden_units, Q_hidden_units)
    agent = agents.RVIQLearningBasedAgent(
        buffer_maxlen, batchsize, env.get_actions(discretization_steps),
        Q, q_lr, rho_lr,
        eps=eps,
        enable_cuda=enable_cuda,
        grad_clip_radius=grad_clip_radius,
        rho_clip_radius=rho_clip_radius)

    if loading_checkpoint:
        agent.load_models(checkpoint_filename)


    end_values = []
    for i in range(num_episodes):
        t0 = time()
        average_action = 0.5*np.ones(2)
        for _ in range(episode_len):
            action = agent.sample_action(env.state)
            average_action = np.mean([average_action, action], axis=0)
            state, reward_cost_tuple, proceed, _  = env.step(action)
            agent.update(reward_cost_tuple, state)
        end_values.append(env.state[0])
        print('Episode {:<6} | ${:>15.2f} | {:.2f}s | {}'.format(i,
                                                        env.state[0],
                                                        time() - t0,
                                                        average_action))
        env.reset()

        if i % checkpoint_interval == 0:
            agent.save_models(os.path.join(experiment_dir, 'models.pt'))
            utils.save_object(end_values, os.path.join(experiment_dir, 'end_values.pkl'))




# end
