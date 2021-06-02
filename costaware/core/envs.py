import gym
import numpy as np
import os
from copy import deepcopy
from gym import spaces
from itertools import product


import costaware.utils.defaults as defaults
from costaware.utils.utils import module_from_path


# Synthetic cost-aware MDP environment

def dict_to_fun(dictionary):
    """
    Wraps a function around a dictionary.

    Parameters
    ----------

    dictionary: a dict

    Returns
    -------
    f: a function

    f(a,b,c,...) == X if and only if dictionary[(a,b,c,...)] == X
    """
    if callable(dictionary):
        return dictionary
    else:
        return lambda *keys: dictionary[keys]


class MDPEnv(gym.Env):
    """
    A Gym-compatible environment that takes fully-specified MDPs.
    """
    
    def __init__(self, states, actions, transition_probabilities, rewards, costs):
        """
        Parameters
        ----------
        states:                   a list of states 
        actions:                  a list of actions (*descriptions* of actions)
        transition_probabilities: a dictionary that returns a state distribution
                                  for a given (state, action) pair
        rewards:                  a dictionary that returns a reward for a given
                                  (state, action) pair
        costs:                    a dictionary that returns a cost for a given
                                  (state, action) pair
        """
        self.states                   = {s: i for i, s in enumerate(states)}
        self.actions                  = {a: i for i, a in enumerate(actions)}
        self.rewards                  = dict_to_fun(rewards)
        self.costs                    = dict_to_fun(costs)
        self.transition_probabilities = dict_to_fun(transition_probabilities)

        self.observation_space = gym.spaces.Discrete(len(states))
        self.action_space      = gym.spaces.Discrete(len(actions))

    def step(self, action):
        """
        Parameters
        ----------
        action: an element of the action space
        """
        action = self.actions[action]
        reward, cost = self.rewards(self.state, action), self.costs(self.state, action)
        distribution = self.transition_probabilities(self.state, action)
        self.state = np.random.choice(self.observation_space.n, p=distribution)

        return self.state, (reward, cost), False, {}

    def reset(self):
        """
        """
        self.state = self.observation_space.sample()


class RandomMDPEnv(MDPEnv):

    def __init__(self, num_states, num_actions, rewards, costs,
                 transition_seed=None, training_seed=None):
        
        np.random.seed(transition_seed)
    
        states  = [s for s in range(num_states)]
        actions = [a for a in range(num_actions)]
    
        probs = {}
        for elem in product(states, actions):
            dist = np.random.random(num_states)
            probs[elem] = dist / np.sum(dist)
    
        def transition_matrix(state, action):
            return probs[(state, action)]
    
        if isinstance(rewards, str):
            rewards = defaults.__dict__[rewards]

        if isinstance(costs, str):
            costs = defaults.__dict__[costs]
    
        np.random.seed(training_seed)
    
        super().__init__(
            states, 
            actions, 
            transition_matrix, 
            rewards, 
            costs
        )


# Obtain the mountain_car module so we can subclass MountainCarEnv
mount_car_file = deepcopy(*gym.envs.__path__) + '/classic_control/mountain_car.py'
mountain_car_path = os.path.abspath(mount_car_file)
MountainCarEnv = module_from_path('mountain_car', mountain_car_path).MountainCarEnv

acrobot_file = deepcopy(*gym.envs.__path__) + '/classic_control/acrobot.py'
acrobot_path = os.path.abspath(acrobot_file)
AcrobotEnv = module_from_path('acrobot', acrobot_path).AcrobotEnv

pendulum_file = deepcopy(*gym.envs.__path__) + '/classic_control/pendulum.py'
pendulum_path = os.path.abspath(pendulum_file)
PendulumEnv = module_from_path('pendulum', pendulum_path).PendulumEnv

cartpole_file = deepcopy(*gym.envs.__path__) + '/classic_control/cartpole.py'
cartpole_path = os.path.abspath(cartpole_file)
CartPoleEnv = module_from_path('cartpole', cartpole_path).CartPoleEnv


class MountainCarCostAwareEnv(MountainCarEnv):
    """
    Extension of the OpenAI MountainCarEnv to include a cost as well as
    a reward. Used to test RL algorithms that maximize the ratio of long-run
    average reward over long-run average cost.

    User can pass in a cost function that takes the current state as input
    and outputs a corresponding cost. Default cost function is 1.0, reducing
    to the reward-only case.
    """

    def __init__(self, goal_velocity=0, cost_fn=lambda x: 1.0):
        MountainCarEnv.__init__(self, goal_velocity)
        self.cost_fn = cost_fn

    def step(self, action):
        state, reward, done, d = MountainCarEnv.step(self, action)
        return state, (reward, self.cost_fn(state)), done, d

    def get_ob(self):
        return np.array(self.state)


class MountainCarCostAwareEnvPositiveReward(MountainCarCostAwareEnv):
    """
    Version of the above with specific cost and altered reward.

    For use in specific trials, should be removed in public release.
    """

    def __init__(self, cost_fn):

        super().__init__(cost_fn=cost_fn)

    def step(self, action):
        state, rc_tuple, done, d = MountainCarCostAwareEnv.step(self, action)
        rc_tuple = -rc_tuple[0], rc_tuple[1]
        return state, rc_tuple, done, d


class AcrobotCostAwareEnv(AcrobotEnv):
    """
    Extension of the OpenAI Acrobot to include a cost as well as
    a reward. Used to test RL algorithms that maximize the ratio of long-run
    average reward over long-run average cost.

    User can pass in a cost function that takes the current state as input
    and outputs a corresponding cost. Default cost function is 1.0, reducing
    to the reward-only case.
    """

    def __init__(self, cost_fn=lambda x: 1.0):
        AcrobotEnv.__init__(self)
        self.cost_fn = cost_fn
    
    def step(self, action):
        state, reward, done, d = AcrobotEnv.step(self, action)
        return state, (reward, self.cost_fn(state)), done, d

    def get_ob(self):
        return AcrobotEnv._get_ob(self)


class PendulumCostAwareEnv(PendulumEnv):
    """
    Extension of the OpenAI PendulumEnv to include a cost as well as
    a reward. Used to test RL algorithms that maximize the ratio of long-run
    average reward over long-run average cost.

    User can pass in a cost function that takes the current state as input
    and outputs a corresponding cost. Default cost function is 1.0, reducing
    to the reward-only case.
    """

    def __init__(self, cost_fn=lambda x: 1.0):
        PendulumEnv.__init__(self)
        self.cost_fn = cost_fn

    def step(self, action):
        state, reward, done, d = PendulumEnv.step(self, action)
        return state, (reward, self.cost_fn(state)), done, d

    def get_ob(self):
        return PendulumEnv._get_obs(self)


class CartPoleCostAwareEnv(CartPoleEnv):
    """
        Extension of the OpenAI CartPoleEnv to include a cost as well as
        a reward. Used to test RL algorithms that maximize the ratio of long-run
        average reward over long-run average cost.

        User can pass in a cost function that takes the current state as input
        and outputs a corresponding cost. Default cost function is 1.0, reducing
        to the reward-only case.

        * this environment returns +1 for each step that is not termination
        """

    def __init__(self, cost_fn=lambda x: 1.0):
        CartPoleEnv.__init__(self)
        self.cost_fn = cost_fn

    def step(self, action):
        state, reward, done, d = CartPoleEnv.step(self, action)
        return state, (reward, self.cost_fn(state)), done, d

    def get_ob(self):
        return np.array(self.state)
