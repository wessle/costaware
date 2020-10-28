import numpy as np
import wesutils
from time import time
from gym.spaces import Discrete

from main.core import agents
from main.experimental.experimental_envs import MountainCarCostAwareEnv

# network and agent parameters
Q_hidden_units = 256
buffer_maxlen = 100000
batchsize = 256
q_lr = 0.001
rho_lr = 0.0001
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
def cost_fn(state):
    cost = max(state[0] + 0.7, 0.1) ** 2
    return cost


# TODO NOT working
def cost_fn1(state):
    position = state[0]
    if position >= 0:
        cost = (position+1.3) ** 2
    else:
        cost = 0.1**2
    return cost


if __name__ == '__main__':

    # create env
    env = MountainCarCostAwareEnv(cost_fn=cost_fn)
    env.reset()

    # gather info about the env
    state_dim = len(env.state)
    num_actions = env.action_space.n
    action_dim = 1 if isinstance(env.action_space, Discrete) else 0

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

    # run the experiment
    end_values, rhos = [], []
    for i in range(1, num_episodes + 1):
        rewards, costs = [], []
        t0 = time()
        for _ in range(episode_len):
            action = agent.sample_action(env.state)
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
        print('ep | rew | cost | time | rho | val_est | Vsref :  '
              '{:<3} | {:>3.2f} | {:>3.2f} | {:.2f}s | {:.4f} | {:.8f} | {:.8f}'.format(
            i, *end_values[-1], time() - t0,
            rhos[-1], agent.ref_val_est, agent.ref_state_val()))

        env.reset()
