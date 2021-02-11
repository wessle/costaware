import numpy as np
import wesutils
from time import time

from main.core import agents
from main.experimental.experimental_envs import PendulumCostAwareEnv


# network and agent parameters
Q_hidden_units = 64
buffer_maxlen = 100000
batchsize = 64
q_lr = 0.001
rho_lr = 0.0001
eps = 0.05
enable_cuda = False
rho_init = 0
grad_clip_radius = None
rho_clip_radius = None

# experiment parameters
num_episodes = 100
episode_len = 300


# Define a cost function to be used in our cost-aware environment
def cost_fn(state):
    """
    state = [cos(theta), sin(theta), ang_vel]
    """
    cost1 = 1 + state[0] / 2
    cost2 = 1 - abs(state[2] / 9)
    # give cost based on the theta value and the angular velocity
    # if the theta is ~1 (standing vertical)
    # give higher cost for smaller angular velocity
    return (0.6*cost1 + 0.4*cost2)**3


if __name__ == '__main__':

    # create env
    env = PendulumCostAwareEnv(cost_fn=cost_fn)
    env.reset()

    # gather info about the env
    state_dim = env.observation_space.shape[0]
    num_actions = len(env.action_space.shape)
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
            action = [action]
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
                'ep', 'rew', 'cost', 'time(s)', 'rho', 'val_est', 'Vsref'))
        print(fmt_vals.format(i, *end_values[-1], time() - t0,
                              rhos[-1], agent.ref_val_est, agent.ref_state_val()))

        env.reset()
