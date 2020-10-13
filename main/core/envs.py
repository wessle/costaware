import gym
import numpy as np
from gym import spaces

from main.utils import moments, ecdf


# Portfolio optimization environments

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
        Gets a discretized representation of the action space.

        Generates an array of uniformly spaced grid points on the unit simplex in a
        given dimension and under a specified level of coarseness.

        The generation is completely deterministic; given a fixed input
        specification, the function always returns the same result.

        This solution always selects the simplex vertices of the form

            v_i = (0, ..., 0, i, 0, ..., 0)  # 1 in the ith coordinate

        and computes the spacing so that the remaining points lie on a uniform
        lattice between the vertices.

        In full generality, the solution is necessarily primitive recursive.

        Params
        ------
        dimension:    float
                      The number of coordinates for each point. In other words, the
                      dimension of the space in which the simplex inhabits. E.g.,
                      the unit simplex in R^3, the triangle, corresponds to
                      dimension=3.
        steps:        int
                      The number of points to be spaced evenly between vertices of
                      the unit simplex. This works like numpy.linspace.

        Return
        ------
        simplex_grid: numpy.ndarray
                      A numpy array of grid points spaced uniformly according to the
                      specifications above.
        """
        def kernel(num, terms):
            """
            Recursive kernel function. This computes the set of "compositions" of
            the number `num` with exactly `terms-1` terms.

            For example, the ways to write the number 4 as a sum of 3 nonnegative
            integers are

                0 + 0 + 4 | 1 + 0 + 3 | 2 + 0 + 2 | 3 + 0 + 1 | 4 + 0 + 0 |
                0 + 1 + 3 | 1 + 1 + 2 | 2 + 1 + 1 | 3 + 1 + 0 |
                0 + 2 + 2 | 1 + 2 + 1 | 2 + 2 + 0 |
                0 + 3 + 2 | 1 + 3 + 0 | 
                0 + 4 + 0 |

            This can be generated by running kernel(3, 4).

            Note: The off by one issue is confusing to me but I am trying to avoid
            thinking through the mathematics too carefully.

            Params
            ------
            num:          int
                          The number to be broken down.
            terms:        int
                          The number of integers to break `num` into.

            Returns
            -------
            compositions: list
                          A list of compositions of `num` composed of `terms-1`
                          terms.

            """
            if terms == 0:
                return [(num,)]
            else:
                return [(i, *j) for i in range(num+1) for j in kernel(num-i, terms-1)]

        return np.array(kernel(num_steps, len(self.portfolio)-1)) / num_steps


class OmegaCostAwareEnv(CostAwareEnv):

    def __init__(self, portfolio, theta):
        self.theta = theta
        super().__init__(portfolio)
        
        self.__numerator_estimate = None
        self.__denominator_estimate = None
        self.__omega_estimate = None

    def reset(self):
        self.ecdf_estimator = ecdf.ECDFEstimator()
        super().reset()

    def _update_reward_and_cost(self, portfolio_returns):
        _ = self.ecdf_estimator(portfolio_returns)
        self.reward = max(0, portfolio_returns - self.theta)
        self.cost = max(0, self.theta - portfolio_returns)

    @property
    def numerator_estimate(self):
        return self.__numerator_estimate

    @property
    def denominator_estimate(self):
        return self.__denominator_estimate

    @property
    def omega_estimate(self):
        cdf = self.ecdf_estimator(self.theta)

        # bounds for the integration
        lower, upper = self.ecdf_estimator.lower, self.ecdf_estimator.upper
        lower -= 1e-1
        upper += 1e-1

        left_tail = np.linspace(lower, self.theta) if lower < self.theta else np.zeros(1)
        right_tail = np.linspace(self.theta, upper) if upper > self.theta else np.zeros(1)

        self.__numerator_estimate = np.trapz(1. - cdf(right_tail), right_tail)
        self.__denominator_estimate = np.trapz(cdf(left_tail), left_tail)
        self.__omega_estimate = self.__numerator_estimate / self.__denominator_estimate

        return self.__omega_estimate


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
