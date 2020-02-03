import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import simple_test.agents as agents
import simple_test.mdp_env as mdp_env


num_states = 10
num_actions = 10
policy_lr = 0.001
v_lr = 0.001
init_mu_r = 1
init_mu_c = 1
mu_lr = 0.0001
mu_floor = 0.01
grad_clip_radius = None # None for no gradient clipping
seed = 1994

num_steps = 100000
print_interval = 1000
make_plot = True
plot_name = 'ac_test.png'

if __name__ == '__main__':

    np.random.seed(seed)

    # set up the MDP
    states = list(range(num_states))
    actions = list(range(num_actions))

    def r(s,a):
        # return s*a
        return s**2

    def c(s,a):
        # return 1 / max(1, s*a)
        return max(1, a**2)

    probs = {}
    for elem in product(states, actions):
        dist = np.random.random(num_states)
        probs[elem] = dist / np.sum(dist)

    def p(s,a):
        # return list(np.ones(num_states) / num_states)
        return probs[(s,a)]

    env = mdp_env.MDPEnv(states, actions, p, r, c)
    env.reset()
    agent = agents.LinearACAgent(states, actions, policy_lr, v_lr,
                                 init_mu_r, init_mu_c, mu_lr,
                                 mu_floor=mu_floor,
                                 grad_clip_radius=grad_clip_radius)

    ratios = []
    for i in range(1, num_steps+1):
        action = agent.sample_action(env.state)
        next_state, rc_tuple, _, _ = env.step(action)
        agent.update(rc_tuple, next_state)
        rho = agent.mu_r / agent.mu_c
        ratios.append(rho)
        if i % print_interval == 0:
            print('Episode {} (rho, state, action): ({:.2f}, {:.2f}, {})'.format(
                i, rho, agent.state, agent.action))

    if make_plot:
        plt.plot(np.arange(num_steps), np.array(ratios))
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.savefig(plot_name)




# end
