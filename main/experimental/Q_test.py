import numpy as np
import wesutils
from time import time
from gym.spaces import Discrete

from main.core import agents
from main.core import models
from main.experimental.experimental_envs import MountainCarCostAwareEnv


# network and agent parameters
Q_hidden_units = 256
buffer_maxlen = 100000
batchsize = 256
q_lr = 0.01
rho_lr = 0.001
eps = 0.1
enable_cuda = False
rho_init = 0
grad_clip_radius = None
rho_clip_radius = None

# experiment parameters
num_episodes = 100
episode_len = 500


# Define a cost function to be used in our cost-aware environment
# Working cost function
def cost_fn_mountain(state):
    # gets bigger as you go further to the right
    return max(state[0] + 0.7, 0.1) ** 2

def cost_fn_mountain2(state):
    # gets smaller as you go further to the right
    return (0.7 - state[0]) ** 2

def cost_fn_mountain3(state):
    # small cost for being to the right of 0.4, cost of 1 otherwise
    return max(1 * (state[0] < 0.1), 0.01)

def cost_fn_mountain4(state):
    # multiple tiers of decreasing costs as goal state 0.5 is appraoched
    loc = state[0]
    return min(1 - 0.99 * (loc >= 0.5),
               1 - 0.9 * (loc > 0.2))

def cost_fn_cartpole(state):
    angle = state[2]
    position = state[0]
    cost = (abs(angle)*2 + abs(position)/5)**2
    return cost


if __name__ == '__main__':

    # create env
    env = MountainCarCostAwareEnv(cost_fn=cost_fn_mountain4)
    env.reset()

    # gather info about the env
    state_dim = len(env.state)
    num_actions = env.action_space.n
    assert isinstance(env.action_space, Discrete), 'action space must be 1D'
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
    agent.set_reference_state(env.state)

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
            action = agent.sample_action(env.state)
            state, reward_cost_tuple, done, _ = env.step(action)
            reward, cost = reward_cost_tuple
            reward = -reward
            rewards.append(reward)
            costs.append(cost)
            agent.update(reward_cost_tuple, state)
            if done:
                break

        # save info and print update
        end_values.append((np.sum(rewards), np.sum(costs)))
        rhos.append(np.mean(rewards) / np.mean(costs))

        if i % 20 == 0:
            print(fmt.format(
                'ep', 'rew', 'cost', 'time(s)', 'rho', 'val_est', 'Vsref'))
        print(fmt_vals.format(i, *end_values[-1], time() - t0,
            rhos[-1], agent.ref_val_est, agent.ref_state_val()))

        env.reset()