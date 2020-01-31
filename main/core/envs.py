import gym
import main.utils.moments as moments
import main.utils.ecdf as ecdf
import main.utils.utils as utils
import numpy as np

from gym import spaces


class AllocationSimplex(spaces.Box):
    """
    Action space of possible portfolio allocations.
    
    Allocations should sum to 1, but we allow those getting close.
    """

    def __init__(self, num_assets):
        super().__init__(0.0, 1.0, shape=(num_assets,))

    def sample(self):
        """Sample uniformly from the unit simplex."""

        v = sorted(np.random.uniform(size=self.shape[0]-1)).append(1.0)
        allocations = [v[i] if i == 0 else v[i] - v[i-1] for i in range(len(v))]
        return np.array(allocations)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return x.shape == self.shape and np.all(x >= self.low) \
                and np.all(x <= self.high) and np.isclose(np.sum(x), 1.0)

    def __repr__(self):
        return "AllocationSimplex" + str(self.shape[0] + "D")

    def __eq__(self, other):
        return isinstance(other, AllocationSimplex) and self.shape == other.shape


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

        self.observation_space = self._observation_space()
        self.action_space = AllocationSimplex(len(portfolio))

    def _observation_space(self):
        """
        Elements in the environment's observation space are flat numpy arrays.
        The number of elements in the array depends on the number of assets in
        the portfolio and the number of statistics being kept track of within
        each asset, as follows:
            [
                value, asset 1 share, asset 2 share, ... , asset n share, 
                asset 1 statistic 1, ..., asset 1 statistic k,
                ... ,
                asset n statistic 1, ..., asset n statistic k
            ]
        
        The upper bounds on all the statistics is infinity, while the lower
        bounds for a given asset are returned by asset.lower_bounds.
        """

        num_assets = len(self.portfolio)
        num_asset_statistics = len(self.portfolio.assets[0])
        size = 1 + num_assets + num_asset_statistics * num_assets
        upper_bounds = np.inf * np.ones(size)
        lower_bounds = (1 + num_assets)*[0]
        for asset in self.portfolio.assets:
            lower_bounds += asset.lower_bounds
        lower_bounds = np.array(lower_bounds)

        return spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.float64)

    @property
    def reward(self):
        """
        """
        return self.__reward

    @reward.setter
    def reward(self, new_reward):
        self.__reward = new_reward

    @property
    def cost(self):
        """
        """
        return self.__cost

    @cost.setter
    def cost(self, new_cost):
        self.__cost = new_cost

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

    def get_actions(self, num_steps):
        """
        Return a discretization of the action space with num_steps+1
        partitions along each axis.
        """

        return utils.generate_simplex(len(self.portfolio), num_steps)


class SharpeCostAwareEnv(CostAwareEnv):

    def __init__(self, portfolio):
        self.estimator = moments.welford_estimator()
        super().__init__(portfolio)

    def reset(self):
        self.estimator = moments.welford_estimator()
        super().reset()

    def _update_reward_and_cost(self, portfolio_returns):
        self.reward, self.cost = self.estimator(portfolio_returns)


class RevisedSharpeCostAwareEnv(SharpeCostAwareEnv):

    def __init__(self, portfolio):
        super().__init__(portfolio)

    def _update_reward_and_cost(self, portfolio_returns):
        mean_estimate, _ = self.estimator(portfolio_returns)
        self.reward = portfolio_returns
        self.cost = (self.reward - mean_estimate)**2


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

        left_tail = np.linspace(lower, self.theta) if lower < self.theta else np.zeros(1)
        right_tail = np.linspace(self.theta, upper) if upper > self.theta else np.zeros(1)

        self.cost   = np.trapz(cdf(left_tail), left_tail)
        self.reward = np.trapz(1. - cdf(right_tail), right_tail)


class SortinoCostAwareEnv(CostAwareEnv):
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

