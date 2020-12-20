import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import array, eye, where
from numpy.linalg import eig, solve

from itertools import product

sns.set_context('paper')


class CostAwareMDP:

    def __init__(self, states, actions, transition, reward, cost):
        """
        Params
        ------
        states: List
            list of states (descriptive labels)
        actions: List
            list of actions (descriptive labels)
        transition: Function(state, action, state) -> float
            transition kernel given a state-action pair
        reward: Function(state, action) -> float
            reward function
        cost: Function(state, action) -> float
            cost function
        """
        self.states = states
        self.actions = actions
        self.transition = transition
        self.reward = reward
        self.cost = cost

    def modified_reward(self, rho, state, action):
        """
        Params
        ------
        rho: float
            ratio threshold for modified CostAware MDP
        state: state
        action: action

        Returns
        -------
        modified_reward: float
            == reward(state, action) - rho * cost(state, action)

        Preconditions
        -------------
        state in self.states
        action in self.actions
        """
        return self.reward(state, action) - rho * self.cost(state, action)

    def stationary_dist(self, policy):
        """
        Computes the stationary distribution for the Cost-Aware MDP under the given
        policy.

        Params
        ------
        policy: dict
            deterministic stationary policy

        Returns
        -------
        dist: np.ndarray
            stationary distribution of the Markov reward process generated by
            the given policy

        Preconditions
        -------------
        all(s in policy.keys() for s in self.states)
        all(policy[s] in self.actions for s in self.states)
        """
        tr_mat = array([
            [self.transition(s,policy[s],t) for t in self.states] \
            for s in self.states
        ])
        evals, evecs = eig(tr_mat)
        
        # get index of stationary distribution
        i = where(abs(evals - 1) < 0.01)[0][0]  
        dist = evecs.T[i]

        # compute the stationary distribution
        dist /= dist.sum()  
    
        return dist

    def _lr_fun(self, policy, fun):
        """
        Computes the long-run average value of the given function under the
        given policy

        Params
        ------
        policy: dict
            deterministic stationary policy
        fun: Function(state, action) -> float
            function to be evaluated in the long run

        Returns
        -------
        lr_fun: float
            long-run average value of the function in the Markov reward process
            generated by the given policy

        Preconditions
        -------------
        all(s in policy.keys() for s in self.states)
        all(policy[s] in self.actions for s in self.states)
        """
        # get the stationary distribution
        dist = self.stationary_dist(policy)
        return sum(p * fun(s,policy[s]) for p, s in zip(dist, self.states))

    def lr_reward(self, policy):
        """
        Computes the long-run average reward of the given policy

        Params
        ------
        policy: dict
            deterministic stationary policy

        Returns
        -------
        lr_reward: float
            long-run average reward of the Markov reward process generated by
            the given policy

        Preconditions
        -------------
        all(s in policy.keys() for s in self.states)
        all(policy[s] in self.actions for s in self.states)
        """
        return self._lr_fun(policy, self.reward)

    def lr_cost(self, policy):
        """
        Computes the long-run average cost of the given policy

        Params
        ------
        policy: dict
            deterministic stationary policy

        Returns
        -------
        lr_cost: float
            long-run average cost of the Markov reward process generated by
            the given policy

        Preconditions
        -------------
        all(s in policy.keys() for s in self.states)
        all(policy[s] in self.actions for s in self.states)
        """
        return self._lr_fun(policy, self.cost)

    def lr_average(self, policy, rho):
        """
        Computes the long-run average modified reward of the given policy for
        the given ratio

        Params
        ------
        policy: dict
            deterministic stationary policy
        rho: float
            specified ratio

        Returns
        -------
        lr_average_reward: float
            long-run average modified reward of the Markov reward process
            generated by the given policy under the specified ratio

        Preconditions
        -------------
        all(s in policy.keys() for s in self.states)
        all(policy[s] in self.actions for s in self.states)
        """
        modified = lambda s, a: self.modified_reward(rho, s, a)
        return self._lr_fun(policy, modified)

    def policy_transition(self, policy):
        """
        Compute the transition matrix associated with the given policy for the
        generated Markov reward process.

        Params
        ------
        policy: dict
            deterministic stationary policy

        Returns
        -------
        matrix: np.ndarray
            2d square stochastic matrix with transition probabilities generated
            by the policy

        Preconditions
        -------------
        all(s in policy.keys() for s in self.states)
        all(policy[s] in self.actions for s in self.states)

        """
        return array([
            [self.transition(s,policy[s],t) for t in self.states] \
            for s in self.states
        ])

    def policies(self):
        """
        Generator for every deterministic stationary policy possible for the
        CostAwareMDP

        Returns
        -------
        policies: Generator[Dict[state, action]]
            a generator of all deterministic stationary policies corresponding
            to the MDP.
                
        Postconditions
        --------------
        len(list(policies)) == len(self.actions) ** len(self.states)
        """
        for actions in product(self.actions, repeat=len(self.states)):
            yield {s: a for s, a in zip(self.states, actions)}

    def value_fun(self, policy, rho):
        """
        Returns the value function corresponding to the given policy and ratio
        rho

        Params
        ------
        policy: Dict[state, action]
            deterministic stationary policy
        rho: float
            specified ratio

        Returns
        -------
        values: Dict[state, float]
            the value function of the modified MDP corresponding to the given
            value of rho for the given policy

        Preconditions
        -------------
        all(s in policy.keys() for s in self.states)
        all(policy[s] in self.actions for s in self.states)
        """

        transition = self.policy_transition(policy)

        reward = array([self.reward(s,policy[s]) for s in self.states]) 
        reward -= self.lr_average(policy,rho)

        value = solve(eye(len(self.states)) - transition, reward) # numpy
        return {s: v for s, v in zip(self.states, value)}

    def qfun(self, policy, rho):
        """
        Returns the Q-function corresponding to the given policy and ratio
        rho

        Params
        ------
        policy: Dict[state, action]
            deterministic stationary policy
        rho: float
            specified ratio

        Returns
        -------
        qfun: Dict[(state, action), float]
            the Q-function of the modified MDP corresponding to the given
            value of rho for the given policy

        Preconditions
        -------------
        all(s in policy.keys() for s in self.states)
        all(policy[s] in self.actions for s in self.states)
        """
        value = self.value_fun(policy, rho)

        q = {
            (s,a): 
            self.modified_reward(rho, s,a) + sum(
                self.transition(s,a,t) * \
                value[t] - self.lr_average(policy, rho) \
                for t in self.states
            ) for s,a in product(self.states, self.actions)
        }

        return q

    def opt_qfun(self, rho):
        """
        Returns the Q-function corresponding to the optimal policy for the given
        ratio rho

        Params
        ------
        rho: float
            specified ratio

        Returns
        -------
        qfun: Dict[(state, action), float]
            the Q-function of the modified MDP corresponding to the given
            value of rho for the optimal policy
        """
        performance_measure = lambda pol: self.lr_average(pol, rho)

        # find the best policy against the performance measure
        optimal_policy = max(self.policies(), key=performance_measure)
        print(optimal_policy)

        # compute its Q-function
        qfun = self.qfun(optimal_policy, rho)

        return qfun


# Forming the CostAwareMDP
states, actions, epsilon = [0, 1], ['L', 'R'], 5e-1

# compute the reward and cost functions
def get_fun(states, actions, values):
    _dict = {
        (s,a): values[i] for i, (s,a) in enumerate(product(states, actions))
    }
    return lambda s, a: _dict[(s,a)]

_r = [0., 1., 1., 0.]
reward = get_fun(states, actions, _r)

_c = [0., 1., -1., 1.]
cost = get_fun(states, actions, _c)

# compute the transition kernel
_transition_dict = {
    ('L',0): 1-epsilon, 
    ('L',1): epsilon, 
    ('R',0): epsilon, 
    ('R',1): 1-epsilon
}
transition = lambda s, a, t: _transition_dict[(a, t)]

camdp = CostAwareMDP(states, actions, transition, reward, cost)


# Plot the discontinuity of Q by coordinate
# rho_vals = np.linspace(-1, 1.5, num=150)
# coords = array([camdp.opt_qfun(t) for t in rho_vals])
# 
# 
# fig, ax = plt.subplots(figsize=(6,3))
# 
# for (s, a), series in zip(product(camdp.states, camdp.actions), coords.T):
#     disconts = np.where(np.abs(np.diff(series) >= 0.2))[0] + 1
#     rhos = rho_vals[:]
#     for i, d in enumerate(disconts):
#         rhos = np.insert(rhos, d+i, rhos[d+i])
#         series = np.insert(series, d+i, np.nan)
# 
#     ax.plot(rhos, series, label=f'$Q^\\rho({s},{a})$')
# 
# ax.set_xlabel(r'$\rho$')
# ax.set_ylabel(r'Coordinate value')
# ax.legend()
# sns.despine(ax=ax)
# # ax.set_title(r"Discontinuity of the function $Q^\rho$")
# 
# fig.savefig('counterexample-qfun.png', bbox_inches='tight', transparent=True)
# plt.close()
