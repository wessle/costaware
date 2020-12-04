import numpy as np
from numpy import cos, sin
import wesutils
from time import time

from main.core import agents
from main.experimental.experimental_envs import AcrobotCostAwareEnv

# network and agent parameters
Q_hidden_units = 256
buffer_maxlen = 100000
batchsize = 128
q_lr = 0.001
rho_lr = 0.0001
eps = 0.1
enable_cuda = False
rho_init = 0
grad_clip_radius = None
rho_clip_radius = None

# experiment parameters
num_episodes = 100
episode_len = 300


# 11/17 learns in ~10 episodes (best rw = -74)
# not very stable
def cost_fn(state):
    """
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    swung height = -cos(state[0]) - cos(state[1] + state[0])
    """
    height = -cos(state[0]) - cos(state[1] + state[0])
    if state[0] > 0:
        # First line is below horizontal, maximize the second angle
        # (between first and second link)
        cost = max(1-state[2], 0.1) ** 2
    else:
        # first link is above horizontal, give higher cost
        # for higher height reached
        cost = (1+height) ** 2
    return cost


# 11/17 learns in ~20 episodes
def cost_fn1(state):
    height = -cos(state[0]) - cos(state[1] + state[0])
    # Height > 0, give greater cost for higher point
    if height > 0:
        cost = (max(1 + height, 1.2))**2
    # OW give cost according to first angle
    else:
        cost = (1 - (state[0]/5))**2
    return cost


# 11/17 learns in ~20 episodes
def cost_fn1(state):
    height = -cos(state[0]) - cos(state[1] + state[0])
    # Height > 0, give greater cost for higher point
    if height > 0:
        cost = (max(1 + height, 1.2))**2
    # OW give cost according to first angle
    else:
        cost = (1 - (state[0]/5))**2
    return cost


if __name__ == '__main__':

    # create env
    env = AcrobotCostAwareEnv(cost_fn = cost_fn)
    env.reset()

    # gather info about the env
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    assert np.ndim(env.action_space.shape) == 1, 'action space must be 1D'
    action_dim = 1

    # create Q function and agent
    Q = wesutils.two_layer_net(state_dim + action_dim, 1,
                               Q_hidden_units, Q_hidden_units)
    agent = agents.DeepRVIQLearningBasedAgent(
        buffer_maxlen, batchsize, np.arange(3),
        Q, q_lr, rho_lr,
        eps=eps,
        enable_cuda=enable_cuda,
        rho_init=rho_init,
        grad_clip_radius=grad_clip_radius,
        rho_clip_radius=rho_clip_radius)
    agent.set_reference_state(env.get_ob())

    # create formats for printing output
    fmt = '{:^5s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s} | {:^10s}'
    fmt_vals = '{:^5} | {:^10.2f} | {:^10.2f} | {:^10.2f} | {:^10.2f} | ' + \
            '{:^10.2f} | {:^10.2f}'

    # run the experiment
    end_values, rhos = [], []
    for i in range(num_episodes):
        rewards, costs = [], []
        t0 = time()
        for _ in range(episode_len):
            action = agent.sample_action(env.get_ob())
            state, reward_cost_tuple, done, _ = env.step(action)
            reward, cost = reward_cost_tuple
            rewards.append(reward)
            costs.append(cost)
            agent.update(reward_cost_tuple, state)
            if done:
                break

        # safe info and print update
        end_values.append((np.sum(rewards), np.sum(costs)))
        rhos.append(np.mean(rewards) / np.mean(costs))

        if i % 20 == 0:
            print(fmt.format(
                'ep', 'rew', 'cost', 'time(s)', 'rho', 'val_est', 'Vsref'))
        print(fmt_vals.format(i, *end_values[-1], time() - t0,
            rhos[-1], agent.ref_val_est, agent.ref_state_val()))

        env.reset()
