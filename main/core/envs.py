import gym
import main.utils.moments as moments
import main.utils.ecdf as ecdf
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
        return self.__reward

    @property
    def cost(self):
        """
        """
        return self.__cost

    def _update_reward_and_cost(self, portfolio_returns):
        raise NotImplementedError("Implemented by subclasses.")

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        self.portfolio.weights = action

        old_value = self.portfolio.value
        self.state = self.portfolio.step()
        new_value = self.portfolio.value

        self._update_reward_and_cost(
            (new_value / old_value) - 1.
        )

        reward, cost = self.reward, self.cost

        return self.state, (reward, cost), False, {}

    def reset(self):
        self.__reward = 0.
        self.__cost   = 0.
        self.portfolio.reset()
        self.state = self.portfolio.summary


class SharpeCostAwareEnv(CostAwareEnv):

    def __init__(self, portfolio):
        self.estimator = moments.welford_estimator()
        super().__init__(portfolio)

    def reset(self):
        self.estimator = moments.welford_estimator()
        super().reset()

    def _update_reward_and_cost(self, portfolio_returns):
        self.__reward, self.__cost = self.estimator(portfolio_returns)


class OmegaCostAwareEnv(CostAwareEnv):
    def __init__(self, portfolio, theta):
        self.theta = theta
        super().__init__(portfolio)

    def reset(self):
        self.ecdf_estimator = ecdf.ECDFEstimator()
        super().reset()

    def _update_reward_and_cost(self, portfolio_returns):
        cdf = self.ecdf_estimator(portfolio_returns)

        # bounds for the integration
        lower, upper = self.ecdf_estimator.lower, self.ecdf_estimator.upper
        lower -= 1e-1
        upper += 1e-1

        # TODO worry about the case where lower > self.theta or upper <
        # self.theta

        left_tail = np.linspace(lower, self.theta)
        right_tail = np.linspace(self.theta, upper)

        self.__cost   = np.trapz(cdf(left_tail), left_tail)
        self.__reward = np.trapz(1. - cdf(right_tail), right_tail)


class SortinoCostAwareEnv(CostAwareEnv):
    def __init__(self, portfolio, threshold):
        self.threshold = threshold
        super().__init__(portfolio)

    def reset(self):
        self.ecdf_estimator = ecdf.ECDFEstimator()
        super().reset()

    def _update_reward_and_cost(self, portfolio_returns):
        cdf = self.ecdf_estimator(portfolio_returns)

        returns_diff = self.threshold - self.ecdf_estimator.values

        self.__reward = portfolio_returns - self.threshold
        self.__cost = np.sqrt(
            np.sum(returns_diff * returns_diff) / len(self.ecdf_estimator)
        )

