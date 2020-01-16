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
        The observation space for the environment is a 1-dimensional numpy array
        whose length depends on the size of the portfolio specified for the
        environment.

        Each asset in the portfolio has four characteristics:
            - price (lower bound = 0)
            - momentum (lower bound = 0)
            - lower bollinger band (lower bound = -oo)
            - upper bollinger band (lower bound = -oo)
        The overall portfolio has two additional characteristics:
            - value (lower bound = 0)
            - shares (lower bound = 0)
        The structure of the observation space is a type of Box space. The order
        is asserted to be
            [
                value, asset 1 share, asset 2 share, ... , asset n share, 
                asset 1 price, asset 1 momentum, asset 1 lower bb, asset 1 upper bb,
                asset 2 price, asset 2 momentum, asset 2 lower bb, asset 2 upper bb,
                ... ,
                asset n price, asset n momentum, asset n lower bb, asset n upper bb,
            ]
        """
        size = 1 + len(self.portfolio) + 4 * len(self.portfolio)
        upper_bounds = np.inf * np.ones(size)
        lower_bounds = np.zeros(size)
        lower_bounds[3+len(self.portfolio)::4] = -np.inf
        lower_bounds[4+len(self.portfolio)::4] = -np.inf

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

