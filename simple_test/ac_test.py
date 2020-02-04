import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import simple_test.agents as agents
import simple_test.mdp_env as mdp_env
import simple_test.functions as functions
import main.utils.utils as utils


config_path = 'ac_config.yml'


if __name__ == '__main__':

    config = utils.load_config(config_path)

    # create variables for all entries in the config file
    for k, v in config.items():
        exec(k + '= v')

    experiment_dir = utils.create_logdir(log_dir, algorithm,
                                         env_name, config_path)

    np.random.seed(transition_seed) if transition_seed is not None \
        else np.random.seed()

    states = list(range(num_states))
    actions = list(range(num_actions))

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

    env = mdp_env.MDPEnv(states, actions, p, r, c)
    env.reset()
    agent = agents.LinearACAgent(states, actions, policy_lr, v_lr,
                                 init_mu_r, init_mu_c, mu_lr,
                                 mu_floor=mu_floor,
                                 grad_clip_radius=grad_clip_radius)

    ratios = []

    if mc_testing:
        time_in_goal = 0
        percentage_time_in_goal = []

    for i in range(1, num_steps+1):
        action = agent.sample_action(env.state)
        next_state, rc_tuple, _, _ = env.step(action)
        agent.update(rc_tuple, next_state)
        rho = agent.mu_r / agent.mu_c
        ratios.append(rho)

        if mc_testing:
            time_in_goal += r(env.state, action)

        if i % print_interval == 0:
            if not mc_testing:
                print('Timestep {} (rho, state, action): ({:.2f}, {}, {})'.format(
                    i, rho, agent.state, agent.action))
            else:
                print('Timestep {} (rho, state, action, goal): ({:.2f}, {}, {}, {:.2f})'.format(
                    i, rho, agent.state, agent.action, time_in_goal / print_interval))
                percentage_time_in_goal.append(time_in_goal / print_interval)
                time_in_goal = 0

            if logging:
                np.save(os.path.join(experiment_dir, 'ratios.npy'), ratios)

                if mc_testing:
                    np.save(os.path.join(experiment_dir, 'percent_time_in_goal.npy'),
                            percentage_time_in_goal)

    if logging:
        plt.plot(np.arange(i), np.array(ratios))
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.savefig(os.path.join(experiment_dir, 'ratios.png'))




# end
