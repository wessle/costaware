import gym
import main.utils.moments as moments
import numpy as np

from gym import spaces


class CostAwareEnv(gym.Env):
    """
    The states and actions are described as follows.

    A state is, fundamentally, a portfolio of financial assets. 

        P = ({A1, A2, ..., An}, value)
    
    At each snapshot, P records the price of each of its component assets Aj as
    well as some other finacial metrics (e.g., momentum and volatility) for
    them. 

    An action is an element w of the unit simplex in Rn. This corresponds to the
    weight of the portfolio's value invested in each asset.
    """

    def __init__(self, portfolio):
        self.portfolio = portfolio

        self.observation_space = spaces.Dict({
            "value": spaces.Box(low=0., high=np.inf, shape=(1,)),
            "shares": spaces.Box(low=0, high=np.inf, shape=(len(portfolio),)),
            "assets": spaces.Dict({
                "price": spaces.Box(
                    low=0, 
                    high=np.inf, 
                    shape=(len(portfolio),)
                ),
                "momentum": spaces.Box(
                    low=0, 
                    high=np.inf,
                    shape=(len(portfolio),)
                ),
                "bollinger": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(2, len(portfolio)))
            })
        })
        self.action_space = spaces.Box(0., 1., shape=(len(portfolio),))

    @property
    def reward(self):
        """
        """
        raise NotImplemented("Reward function must be implemented by subclasses.")

    @property
    def cost(self):
        """
        """
        raise NotImplemented("Cost function must be implemented by subclasses.")

    def step(self, action):
        """
        
        """
        raise NotImplemented("step function must be implemented by subclasses.")
        
    def reset(self):
        self.portfolio.reset()
        self.state = self.portfolio.summary


class SharpeCostAwareEnv(CostAwareEnv):

    def __init__(self, portfolio):
        self.estimator = moments.welford_estimator()
        super().__init__(portfolio)

    def reset(self):
        self.__reward = 0.
        self.__cost   = 0.
        self.estimator = moments.welford_estimator()
        super().reset()

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        self.portfolio.weights = action

        old_value = self.portfolio.value
        self.state = self.portfolio.step()
        new_value = self.portfolio.value

        self.__reward, self.__cost = self.estimator(
            (new_value / old_value) - 1.
        )

        reward, cost = self.reward, self.cost

        return self.state, (reward, cost), False, {}

    @property
    def reward(self):
        return self.__reward

    @property
    def cost(self):
        return self.__cost
