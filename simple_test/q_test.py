import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import simple_test.agents as agents
import simple_test.mdp_env as mdp_env
import main.utils.utils as utils


config_path = 'q_config.yml'

###### Reward and cost functions for use in testing

def r(s,a):
    # return s*a
    return s**2

def c(s,a):
    # return 1 / max(1, s*a)
    return max(1, a**2)


if __name__ == '__main__':

    config = utils.load_config(config_path)

    # create variables for all entries in the config file
    for k, v in config.items():
        exec(k + '= v')

    np.random.seed(seed)

    states = list(range(num_states))
    actions = list(range(num_actions))

    probs = {}
    for elem in product(states, actions):
        dist = np.random.random(num_states)
        probs[elem] = dist / np.sum(dist)

    def p(s,a):
        return probs[(s,a)]

    env = mdp_env.MDPEnv(states, actions, p, r, c)
    env.reset()
    agent = agents.TabularQAgent(states, actions, q_lr, rho_lr,
                                 rho_init=rho_init, eps=eps)

    ratios = []
    for i in range(1, num_steps+1):
        action = agent.sample_action(env.state)
        next_state, rc_tuple, _, _ = env.step(action)
        agent.update(rc_tuple, next_state)
        ratios.append(agent.rho)
        if i % print_interval == 0:
            print('Episode {} (rho, state, action): ({:.2f}, {}, {})'.format(
                i, agent.rho, agent.state, agent.action))

    if logging:
        experiment_dir = utils.create_logdir(log_dir, algorithm,
                                             env_name, config_path)
        np.save(os.path.join(experiment_dir, 'ratios.npy'), ratios)
        plt.plot(np.arange(num_steps), np.array(ratios))
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.savefig(os.path.join(experiment_dir, 'ratios.png'))






# end
