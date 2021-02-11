import numpy as np
import wesutils
from time import time
from gym.spaces import Discrete

from main.core import agents
from main.core import models
from main.experimental.experimental_envs import MountainCarCostAwareEnv, \
        CartPoleCostAwareEnv, PendulumCostAwareEnv, AcrobotCostAwareEnv


# network and agent parameters
net_width = 64
buffer_maxlen = 1000000
batchsize = 256
policy_lr = 0.01
v_lr = 0.1
init_mu_r = 1
init_mu_c = 1
mu_lr = 0.005
enable_cuda = False
grad_clip_radius = None
reward_cost_mean_floor = 1e-8

# experiment parameters
num_episodes = 100
episode_len = 500


# Define a cost function to be used in our cost-aware environment
# Working cost function

def cost_fn_mountain(state):
    return max(state[0] + 0.7, 0.1) ** 2

# def cost_fn_mountain(state):
#     return 10 * max(state[0] + 0.7, 0.1)

def cost_fn_cartpole(state):
    angle = state[2]
    position = state[0]
    cost = (abs(angle)*2 + abs(position)/5)**2
    return cost

def cost_fn_pendulum(state):
    """
    state = [cos(theta), sin(theta), ang_vel]
    """
    cost1 = 1 + state[0] / 2
    cost2 = 1 - abs(state[2] / 9)
    # give cost based on the theta value and the angular velocity
    # if the theta is ~1 (standing vertical)
    # give higher cost for smaller angular velocity
    return (0.6*cost1 + 0.4*cost2)**3

def cost_fn_acrobot(state):
    """
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    swung height = -cos(state[0]) - cos(state[1] + state[0])
    """
    height = -np.cos(state[0]) - np.cos(state[1] + state[0])
    if state[0] > 0:
        # First line is below horizontal, maximize the second angle
        # (between first and second link)
        cost = max(1-state[2], 0.1) ** 2
    else:
        # first link is above horizontal, give higher cost
        # for higher height reached
        cost = (1+height) ** 2
    return cost


if __name__ == '__main__':

    # create env
    env = MountainCarCostAwareEnv(cost_fn=cost_fn_mountain)
    env.reset()

    # gather info about the env
    state_dim = len(env.state)
    num_actions = env.action_space.n
    assert isinstance(env.action_space, Discrete), 'action space must be 1D'
    action_dim = 1

    # create agent
    policy = models.CategoricalPolicyTwoLayer(state_dim, num_actions,
                                              net_width, net_width)
    v_net = wesutils.two_layer_net(state_dim, 1)
    agent = agents.DeepACAgent(
        buffer_maxlen, batchsize,
        policy, v_net,
        policy_lr, v_lr,
        init_mu_r=init_mu_r, init_mu_c=init_mu_c,
        mu_lr=mu_lr,
        enable_cuda=enable_cuda,
        grad_clip_radius=grad_clip_radius,
        reward_cost_mean_floor=reward_cost_mean_floor
    )


    # create formats for printing output
    fmt = '{:^5s} | {:^10s} | {:^10s} | {:^10s} | {:^10s}'
    fmt_vals = '{:^5} | {:^10.2f} | {:^10.2f} | {:^10.2f} | {:^10.2f}'

    # run the experiment
    end_values, rhos = [], []
    for i in range(num_episodes):
        rewards, costs = [], []
        t0 = time()
        for _ in range(episode_len):
            action = agent.sample_action(env.state)[0]
            state, reward_cost_tuple, done, _ = env.step(action)
            reward, cost = reward_cost_tuple[::-1]
            cost = -cost
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
                'ep', 'rew', 'cost', 'time(s)', 'rho'))
        print(fmt_vals.format(i, *end_values[-1], time() - t0,
            rhos[-1]))

        env.reset()
