import gym
import numpy as np


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
        self.action_space      = gym.spaces.Discrete(len(actions[states[0]]))

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
