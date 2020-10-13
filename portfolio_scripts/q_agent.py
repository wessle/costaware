import gym
import numpy as np
import torch
from collections import OrderedDict
from time import time
import os
import sys
import wesutils

import main.utils.utils as utils
from main.core import agents, asset, envs


config_path = 'q_config.yml'

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

    Q = wesutils.two_layer_net(state_dim + action_dim, 1,
                            Q_hidden_units, Q_hidden_units)
    agent = agents.DeepRVIQLearningBasedAgent(
        buffer_maxlen, batchsize, env.get_actions(discretization_steps),
        Q, q_lr, rho_lr,
        eps=eps,
        enable_cuda=enable_cuda,
        rho_init=rho_init,
        grad_clip_radius=grad_clip_radius,
        rho_clip_radius=rho_clip_radius)

    if loading_checkpoint:
        agent.load_models(checkpoint_filename)

    # set reference state to be used in updates
    agent.set_reference_state(env.state)

    end_values = []
    rhos = []
    for i in range(1, num_episodes+1):
        average_action = np.zeros(action_dim)
        t0 = time()
        for _ in range(episode_len):
            action = agent.sample_action(env.state)
            average_action += action
            state, reward_cost_tuple, proceed, _  = env.step(action)
            agent.update(reward_cost_tuple, state)
        end_values.append(env.portfolio.value)
        rhos.append(agent.rho)
        print('Episode {:<6} | ${:>10.2f} | {:.2f}s | {} | {:.4f} | {:.8f}'.format(
            i, env.portfolio.value, time() - t0,
            (average_action / episode_len).round(decimals=2),
            agent.rho, agent.ref_state_val()))

        env.reset()

        if i % checkpoint_interval == 0:
            agent.save_models(os.path.join(experiment_dir, 'models.pt'))
            np.save(os.path.join(experiment_dir, 'end_values.npy'), end_values)
            np.save(os.path.join(experiment_dir, 'rhos.npy'), rhos)




# end
