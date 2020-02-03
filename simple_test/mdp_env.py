import gym

class MDPEnv(gym.Env):
    """
    A Gym-compatible environment that takes fully-specified MDPs.
    """
    
    def __init__(self, states, actions, transition_probabilities, rewards, costs):
        """
        Parameters
        ----------
        states:                   a list of states 
        actions:                  a list of actions
        transition_probabilities: a function that returns a state distribution
                                  for a given (state, action) pair
        rewards:                  a function that returns a reward for a given
                                  (state, action) pair
        costs:                    a function that returns a cost for a given
                                  (state, action) pair
        """
        self.states                   = states
        self.actions                  = actions
        self.rewards                  = rewards
        self.costs                    = costs
        self.transition_probabilities = transition_probabilities

        self.observation_space = gym.spaces.Discrete(len(states))
        self.action_space      = gym.spaces.Discrete(len(actions[states[0]]))


    def step(self, action):
        """
        Parameters
        ----------
        action: an element of the action space
        """
        reward, cost = self.rewards(self.state, action), self.costs(self.state, action)
        distribution = self.transition_probabilities(self.state, action)
        self.state = np.random.choice(self.observation_space.n, p=distribution)

        return self.state, (reward, cost), False, {}

    def reset(self):
        """
        """
        self.state = self.observation_space.sample()
