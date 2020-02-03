import gym

class MDPEnv(gym.Env):
    """
    """
    
    def __init__(self, states, actions, transition_probabilities, rewards, costs):
        """
        Parameters
        ----------
        """
        self.observation_space = gym.spaces.Discrete(len(states))
        self.action_space = 
        pass

    def step(self, action):
        """
        """
        pass

    def reset(self):
        """
        """
        pass

    def seed(self, seed=None):
        """
        """
        pass


