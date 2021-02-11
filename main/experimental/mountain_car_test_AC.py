import numpy as np
import wesutils
from time import time
from gym.spaces import Discrete

from main.core import agents
from main.core import models
from main.experimental.experimental_envs import MountainCarCostAwareEnv


# network and agent parameters
net_width = 16
buffer_maxlen = 100000
batchsize = 64
policy_lr = 0.01
v_lr = 0.01
init_mu_r = 0
init_mu_c = 0
mu_lr = 0.005
enable_cuda = False
grad_clip_radius = None
reward_cost_mean_floor = 1e-8

# experiment parameters
num_episodes = 100
episode_len = 500


# Define a cost function to be used in our cost-aware environment
# Working cost function
def cost_fn(state):
    return max(state[0] + 0.7, 0.1) ** 2


if __name__ == '__main__':

    # create env
    env = MountainCarCostAwareEnv(cost_fn=cost_fn)
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
            reward, cost = reward_cost_tuple
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
