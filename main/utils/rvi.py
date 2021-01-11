from itertools import product
from functools import partial
from collections.abc import Callable, MutableMapping

import numpy as np
from scipy.optimize import linprog

from main.core.envs import MDPEnv, RandomMDPEnv
import main.utils.defaults as defaults


class ValueFn(MutableMapping, Callable):
    """
    Wrapper class for value functions. These objects are glorified dictionaries
    that also have "functional" features (they can be called as functions).
    """

    def __init__(self, states, values=None):
        # the domain attribute is a dictionary with state keys (descriptive) and
        # index values (integers)
        self.domain = {s: i for i, s in enumerate(states)}

        # the actual values are stored in an array to speed up some computations
        self.vals = np.zeros(len(self.domain)) if values is None else values

    def __call__(self, state):
        """
        Wrap calling over dictionary indexing
        """
        return self[state]

    def __getitem__(self, state):
        """
        Get the value for a given state
            state --> index --> value at index
        """
        return self.vals[self.domain[state]]

    def expect(self, dist):
        """
        Computes the expectation
            sum(self.vals[i] * dist[i] for i in self.domain)
        where the assumption is that the distribution is indexed like the states
        """
        return dist @ self.vals

    def __iter__(self):
        """
        Calling iter goes through the states
        """
        return iter(self.domain)

    def __len__(self):
        """
        Number of states
        """
        return len(self.domain)

    def __delitem__(self, state):
        """
        Delete a state from the support of the value function. (Don't use this,
        but it's needed to instantiate a subclass of MutableMapping.)
        """
        self.vals = self.vals[np.arange(len(self)) != self.domain[state]]
        self.domain.__delitem__(state)

    def __setitem__(self, state, val):
        """
        Sets the value function at a given state as specified.
        """
        self.vals[self.domain[state]] = val

    def __repr__(self):
        """
        String representation, just show the array
        """
        return repr(self.vals)


class QFn(Callable, MutableMapping):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.domain = {
            (s,a): (i, j) for (i, s), (j, a) in product(enumerate(states), enumerate(actions))
        }
        self.qvals = np.zeros((len(states), len(actions)))

    def __call__(self, state, action):
        return self.qvals[self.domain[(state, action)]]

    def __getitem__(self, key):
        state, action = key
        return self.qvals[self.domain[(state, action)]]

    def expect(self, dist):
        """
        Computes the expectation
            sum(self.qvals[i] * dist[i] for i in self.domain)
        """
        return self.to_value().expect(dist)

    def to_value(self):
        """
        Returns the derived value function of the given Q function
        """
        return ValueFn(self.states, values=np.max(self.qvals, axis=1))


    def __iter__(self):
        return zip(self.domain)

    def __len__(self):
        return len(self.domain)

    def __delitem__(self, key):
        state, action = key
        index = self.domain[(state, action)]
        self.qvals = self.qvals[np.arange(len(self)) != index]
        self.domain.__delitem__((state, action))

    def __setitem__(self, key, val):
        state, action = key
        self.qvals[self.domain[(state, action)]] = val

    def __repr__(self):
        return repr(self.qvals)


class RVIterator:
    
    def __init__(self, mdp):
        self.mdp = mdp

    def _value_op(self, rho, value_fn, state):
        """
        Compute the internal RVI operator:
            max r(s,a) - rho * c(s,a) - v(0) + sum p(t|s,a) * v(t))
             a                                    t
        """
        def adjusted_reward(action):
            return self.mdp.rewards(state,action) - rho * self.mdp.costs(state,action)

        def expectation(action):
            return value_fn.expect(self.mdp.transition_probabilities(state, action))

        offset = value_fn[self.mdp.states[0]]

        return max(adjusted_reward(a) - offset + expectation(a) for a in self.mdp.actions)

    def _q_op(self, rho, q_fn, state, action):
        """
        Compute the internal RQI operator
            r(s,a) - rho * c(s,a) - Q(0,0) + sum p(t|s,a) * max Q(t,b)
                                              t              b
        """
        adjusted_reward = self.mdp.rewards(state, action) - rho * self.mdp.costs(state,action) 
        expectation = q_fn.expect(self.mdp.transition_probabilities(state, action))
        offset = q_fn[self.mdp.states[0],self.mdp.actions[0]]

        return adjusted_reward - offset  + expectation

    def rvi(self, rho, n_iter=20):
        """
        Relative value iteration for a given ratio rho
        """
        value_fn = ValueFn(self.mdp.states)

        for _, s in product(range(n_iter), self.mdp.states):
            value_fn[s] = self._value_op(rho, value_fn, s)

        return value_fn

    def rqi(self, rho, n_iter=20):
        """
        Relative Q function iteration for a given ratio rho
        """
        q_fn = QFn(self.mdp.states, self.mdp.actions)

        for _, s, a in product(range(n_iter), self.mdp.states, self.mdp.actions):
            q_fn[s,a] = self._q_op(rho, q_fn, s, a)

        return q_fn


class AverageRewardLP:

    def __init__(self, mdp):
        self.mdp = mdp

    def solve(self, rho):
        """
        Compute the average reward value function, Q-function, and optimal policy for 
        a given MDP
        """
        # Formulate the LP
        
        n_vars = 1 + len(self.mdp.states)
        n_constr = len(self.mdp.states) * len(self.mdp.actions)
        
        # Minimize the value of the first variable
        objective = np.zeros(n_vars)
        objective[0] += 1.
        
        constr_vec = np.zeros(n_constr)
        constr_mat = np.zeros((n_constr, n_vars))
        
        # The first variable is added to each of the constraint inequalities
        constr_mat[:,0] = 1.
        
        # Loop through all constraint inequalities...
        for row, (s, a) in enumerate(product(self.mdp.states, self.mdp.actions)):
            constr_vec[row] = self.mdp.rewards(s, a) - rho * self.mdp.costs(s, a)
            
            # Loop through the sum (expectation) in each inequality
            for col, t in enumerate(self.mdp.states, start=1):
                constr_mat[row, col] = int(s == t) - self.mdp.transition_probabilities(s,a)[t]
                    
        # Solve the LP
        result = self._solve_lp(objective, constr_mat, constr_vec) 
    
        # Return the *interpretation* of the LP in our MDP context
        return self.interpret(result, rho)
    
    def _solve_lp(self, c, A, b):
        """
        Solve the LP formulated as 
            min c^Tx
            s.t. Ax >= b
            x unbounded
        """
        result = linprog(c, A_ub=-A, b_ub=-b,bounds=None, method='simplex')
        assert result.status == 0 and result.success, \
            "LP error: {lp_error_dict[result.status]}"
        
        return result

    def interpret(self, result, rho):
        """
        Given the raw result from the LP, produce the relevant functions for 
        average-reward optimality:
            * the value function
            * the Q function
            * an average-reward optimal policy
        """
        offset, rawvalues = result.x[0], result.x[1:]
        value = {s: v for s, v in zip(self.mdp.states, rawvalues)}
        
        policy = self.get_policy(value, offset, rho)
        qfun = self.get_qfun(value, offset, rho)
        
        return value, qfun, policy

    def _optfun(self, rho, value, g, s, a):
        """
        """

        adjusted_reward = self.mdp.rewards(s,a) - rho * self.mdp.costs(s,a)
        expectation = sum(self.mdp.transition_probabilities(s,a)[t] * value[t] for t in self.mdp.states)
        return  adjusted_reward - g - value[s] + expectation
                

    def get_policy(self, value, g, rho):
        """
        Compute a policy corresponding to the average reward value function with 
        offset g
        """
        opt = partial(self._optfun, rho, value, g)
        
        policy = {s: max(self.mdp.actions, key=partial(opt, s)) for s in self.mdp.states}
        
        return policy
    
    def get_qfun(self, value, g, rho):
        """
        Compute a Q-function corresponding to the average reward value function with 
        offset g
        """
        opt = partial(self._optfun, rho, value, g)

        return {
            (s,a): opt(s, a) for s, a in product(self.mdp.states, self.mdp.actions)
        }
