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

    # Set the number of threads pytorch can use
    torch.set_num_threads(config['num_threads'])

    # experiment parameters
    algorithm_name = config['algorithm_name']
    env_name = config['env_name']
    env_target_return = config['env_target_return']
    checkpoint_interval = config['checkpoint_interval']
    log_dir = config['log_dir']
    num_episodes = config['num_episodes']
    episode_len = config['episode_len']
    experiment_dir = utils.create_logdir(log_dir, algorithm_name,
                                         env_name, config_path)
    loading_checkpoint = config['loading_checkpoint']

    # asset parameters
    init_prices = config['init_prices']
    mean_returns = config['mean_returns']
    stdev_returns = config['stdev_returns']
    init_weights = config['init_weights']
    init_principal = config['init_principal']

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


    # agent parameters
    action_dim = len(portfolio)
    state_dim = len(env.state)
    buffer_maxlen = config['buffer_maxlen']
    batchsize = config['batchsize']
    discretization_steps = config['discretization_steps']
    q_lr = config['q_lr']
    rho_lr = config['rho_lr']
    eps = config['eps']
    rho_init = config['rho_init']
    enable_cuda = config['enable_cuda']
    grad_clip_radius = config['grad_clip_radius']
    rho_clip_radius = config['rho_clip_radius']
    Q_hidden_units = config['Q_hidden_units']

    Q = utils.two_layer_net(state_dim + action_dim, 1,
                            Q_hidden_units, Q_hidden_units)
    agent = agents.RVIQLearningBasedAgent(
        buffer_maxlen, batchsize, env.get_actions(discretization_steps),
        Q, q_lr, rho_lr,
        eps=eps,
        enable_cuda=enable_cuda,
        rho_init=rho_init,
        grad_clip_radius=grad_clip_radius,
        rho_clip_radius=rho_clip_radius)

    if loading_checkpoint:
        agent.load_models(checkpoint_filename)

    agent.set_reference_state(env.state)

    end_values = []
    for i in range(num_episodes):
        average_action = np.zeros(action_dim)
        t0 = time()
        for _ in range(episode_len):
            action = agent.sample_action(env.state)
            average_action += action
            state, reward_cost_tuple, proceed, _  = env.step(action)
            agent.update(reward_cost_tuple, state)
        end_values.append(env.state[0])
        print('Episode {:<6} | ${:>10.2f} | {:.2f}s | {} | {:.4f} | {:.8f}'.format(
            i, env.portfolio.value, time() - t0,
            (average_action / episode_len).round(decimals=2),
            agent.rho, agent.ref_state_val()))

#        import pdb; pdb.set_trace()

        env.reset()

        if i + 1 % checkpoint_interval == 0:
            agent.save_models(os.path.join(experiment_dir, 'models.pt'))
            utils.save_object(end_values, os.path.join(experiment_dir, 'end_values.pkl'))




# end
