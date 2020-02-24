import gym
import numpy as np
from gym import spaces

from main.core.envs import CostAwareEnv
from main.utils import moments, ecdf


# Portfolio optimization environments

class SharpeCostAwareExperimentalEnv(CostAwareEnv):

    def __init__(self, portfolio):
        self.estimator = moments.welford_estimator()
        super().__init__(portfolio)

    def reset(self):
        self.estimator = moments.welford_estimator()
        super().reset()

    def _update_reward_and_cost(self, portfolio_returns):
        self.reward, self.cost = self.estimator(portfolio_returns)


class RevisedSharpeCostAwareExperimentalEnv(SharpeCostAwareExperimentalEnv):

    def __init__(self, portfolio):
        super().__init__(portfolio)

    def _update_reward_and_cost(self, portfolio_returns):
        mean_estimate, _ = self.estimator(portfolio_returns)
        self.reward = portfolio_returns
        self.cost = (self.reward - mean_estimate)**2


class OmegaCostAwareExperimentalEnv(CostAwareEnv):
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

        left_tail = np.linspace(lower, self.theta) if lower < self.theta else np.zeros(1)
        right_tail = np.linspace(self.theta, upper) if upper > self.theta else np.zeros(1)

        self.cost   = np.trapz(cdf(left_tail), left_tail)
        self.reward = np.trapz(1. - cdf(right_tail), right_tail)


class RevisedOmegaCostAwareExperimentalEnv(CostAwareEnv):

    def __init__(self, portfolio, theta):
        self.theta = theta
        super().__init__(portfolio)
        
        self.__numerator_estimate = None
        self.__denominator_estimate = None
        self.__omega_estimate = None


class SortinoCostAwareExperimentalEnv(CostAwareEnv):
    def __init__(self, portfolio, threshold):
        self.threshold = threshold
        super().__init__(portfolio)

    def reset(self):
        self.ecdf_estimator = ecdf.ECDFEstimator()
        super().reset()

    def _update_reward_and_cost(self, portfolio_returns):
        cdf = self.ecdf_estimator(portfolio_returns)

        # only count the returns which are less than the threshold in this
        # computation.
        returns_diff = (self.threshold - self.ecdf_estimator.values) *\
            (self.ecdf_estimator.values <= self.threshold)

        self.reward = portfolio_returns - self.threshold
        self.cost = np.sqrt(
            np.sum(returns_diff * returns_diff) / len(self.ecdf_estimator)
        )
