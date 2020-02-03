import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import simple_test.agents as agents
import simple_test.mdp_env as mdp_env


num_states = 20
num_actions = 20
q_lr = 0.01
rho_lr = 0.001
rho_init = 0
eps = 0.05
seed = 1994

num_steps = 4000000
print_interval = 10000
make_plot = True

if __name__ == '__main__':

    np.random.seed(seed)

    states = list(range(num_states))
    actions = list(range(num_actions))

    def r(s,a):
        return s*a

    def c(s,a):
        return 1 / max(1, s*a)

    probs = {}
    for elem in product(states, actions):
        dist = np.random.random(num_states)
        probs[elem] = dist / np.sum(dist)

    def p(s,a):
        # return list(np.ones(num_states) / num_states)
        return probs[(s,a)]

    env = mdp_env.MDPEnv(states, actions, p, r, c)
    env.reset()
    agent = agents.TabularQLearner(states, actions, q_lr, rho_lr,
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

    if make_plot:
        plt.plot(np.arange(num_steps), np.array(ratios))
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.savefig('test.png')
