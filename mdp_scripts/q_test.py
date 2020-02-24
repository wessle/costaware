import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import deque
import wesutils

import main.core.agents as agents
import main.core.envs as envs
import mdp_scripts.functions as functions
import main.utils.utils as utils


config_path = 'q_config.yml'


if __name__ == '__main__':

    config = wesutils.load_config(config_path)

    # create variables for all entries in the config file
    for k, v in config.items():
        exec(k + '= v')

    experiment_dir = wesutils.create_logdir(log_dir, algorithm,
                                         env_name, config_path)

    states = list(range(num_states))
    actions = list(range(num_actions))

    np.random.seed(transition_seed) if transition_seed is not None \
        else np.random.seed()

    probs = {}
    for elem in product(states, actions):
        dist = np.random.random(num_states)
        probs[elem] = dist / np.sum(dist)

    def p(s,a):
        return probs[(s,a)]

    r = eval('functions.' + r)
    c = eval('functions.' + c)
    
    np.random.seed(training_seed) if training_seed is not None \
        else np.random.seed()

    env = envs.MDPEnv(states, actions, p, r, c)
    env.reset()
    agent = agents.TabularQAgent(states, actions, q_lr, rho_lr,
                                 rho_init=rho_init, eps=eps)

    ratios = []

    if mc_testing:
        time_in_goal = deque(maxlen=mc_testing_time_window)
        percentage_time_in_goal = []

    for i in range(1, num_steps+1):
        action = agent.sample_action(env.state)
        next_state, rc_tuple, _, _ = env.step(action)
        agent.update(rc_tuple, next_state)
        ratios.append(agent.rho)

        if mc_testing:
            time_in_goal.append(r(env.state, action))
            percentage_time_in_goal.append(sum(time_in_goal) / len(time_in_goal))

        if i % print_interval == 0:
            if not mc_testing:
                print('Timestep {} (rho, state, action): ({:.2f}, {}, {})'.format(
                    i, agent.rho, agent.state, agent.action))

            else:
                print('Timestep {} (rho, state, action, goal): ({:.2f}, {}, {}, {:.2f})'.format(
                    i, agent.rho, agent.state, agent.action, percentage_time_in_goal[-1]))

            if logging:
                np.save(os.path.join(experiment_dir, 'ratios.npy'), ratios)

                if mc_testing:
                    np.save(os.path.join(experiment_dir, 'percent_time_in_goal.npy'),
                            percentage_time_in_goal)

    if logging:
        plt.plot(np.arange(num_steps), np.array(ratios))
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.savefig(os.path.join(experiment_dir, 'ratios.png'))

        if mc_testing:
            plt.clf()
            plt.plot(np.arange(num_steps), np.array(percentage_time_in_goal))
            plt.xlabel('Step')
            plt.ylabel('% time spent in goal state')
            plt.savefig(os.path.join(experiment_dir, 'percent_time_in_goal.png'))








# end
