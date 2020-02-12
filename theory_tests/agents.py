import numpy as np
import numbers
from scipy.special import softmax
from copy import deepcopy


###### Tabular RVI Q-learning-based agent

class TabularQAgent:
    """RVI Q-learning agent for the tabular case."""

    def __init__(self, states, actions, q_lr, rho_lr,
                 rho_init=0, eps=0.1, ref_state=None):

        self.states = tuple(states)
        self.state_indices = {s: i for i, s in enumerate(states)}
        self.actions = tuple(actions)
        self.action_indices = {a: i for i, a in enumerate(actions)}
        self.q_lr = q_lr
        self.rho_lr = rho_lr
        self.rho = rho_init
        self.eps = eps
        self.__Q_table = np.zeros((len(states), len(actions)))
        self.__ref_state = ref_state if ref_state is not None \
                else self.states[0]

        self.state = None
        self.action = None

    @property
    def Q_table(self):
        return self.__Q_table

    @Q_table.setter
    def Q_table(self, M):
        self.__Q_table = M

    def inplace_add_to_Q_table(self, s, a, val):
        """Perform the operation: Q_table(s,a) += val"""

        self.__Q_table[self.state_indices[s]][self.action_indices[a]] += val

    @property
    def ref_state(self):
        return self.__ref_state

    @ref_state.setter
    def ref_state(self, s):
        self.__ref_state = s

    def Q(self, s, a):
        return self.__Q_table[self.state_indices[s]][self.action_indices[a]]

    def V(self, s):
        return np.max(self.__Q_table[self.state_indices[s]])

    def ref_state_val(self):
        return self.V(self.ref_state)

    def greedy_action(self, s):
        return self.actions[np.argmax(self.__Q_table[self.state_indices[s]])]

    def sample_action(self, s):
        """
        Sample an action epsilon-greedily.
        
        An action must be sampled before self.update can be called.
        """

        self.state = s

        greedy = self.greedy_action(s)
        if np.random.random() < self.eps:
            self.action = np.random.choice(self.actions)
            while self.action == greedy:
                self.action = np.random.choice(self.actions)
        else:
            self.action = greedy

        return self.action

    def update(self, reward_cost_tuple, next_state):
        """Perform the update step."""

        assert self.state is not None, "sample_action must be called first"

        reward, cost = reward_cost_tuple

        td_err = reward - self.rho * cost + self.V(next_state) \
                - self.ref_state_val() - self.Q(self.state, self.action)
        self.inplace_add_to_Q_table(self.state, self.action, self.q_lr * td_err)
        self.rho += self.rho_lr * self.ref_state_val()


###### Actor-critic agent and associated objects

class LinearApproximator:
    """Linear function approximator with scalar output."""
    
    def __init__(self, input_dim, initialization_cov_constant=1):
        self.input_dim = input_dim
        self.initialization_cov_constant = initialization_cov_constant
        self.num_params = self.input_dim + 1
        self.__params = None
        self.reinit_params()
        
    def reinit_params(self):
        self.__params = np.random.multivariate_normal(
            mean=np.zeros(self.num_params),
            cov=self.initialization_cov_constant*np.eye(self.num_params))
        
    @property
    def params(self):
        return self.__params
    
    @params.setter
    def params(self, new_params):
        self.__params = new_params
        
    def gradient(self, x):
        return np.append(x, 1)

    def __call__(self, x):
        x = np.append(x, 1)
        return np.dot(self.params, x)
    
    
class SoftmaxPolicy:
    """"
    Parent class for use in softmax policies with specific
    h functions.
    """
    
    def __init__(self, actions):
        self.actions = actions
        self.action_indices = {a: i for i, a in enumerate(actions)}
        self.h = None
        
    @property
    def params(self):
        return self.h.params

    @params.setter
    def params(self, new_params):
        self.h.params = new_params

    def reinit_params(self):
        self.h.reinit_params()

    def _hvals(self, state):
        return np.array([self.h(np.append(state, action))
                         for action in self.actions])
    
    def _get_probs(self, state):
        return softmax(self._hvals(state))
        
    def sample_action(self, state):
        probs = self._get_probs(state)
        return np.random.choice(self.actions, p=probs)
    
    def _hgrads(self, state):
        return np.array([self.h.gradient(np.append(state, action))
                        for action in self.actions])
        
    def grad_log_policy(self, state, action):    
        hgrads = self._hgrads(state)
        action_index = self.action_indices[action]
        return hgrads[action_index] - np.dot(self._get_probs(state).flatten(), hgrads)
    
    def pdf(self, action, state):
        return self._get_probs(state)[self.action_indices[action]]
    
    
class SoftmaxPolicyLinear(SoftmaxPolicy):
    def __init__(self, state_vector_size, actions,
                 initialization_cov_constant=1):
        
        SoftmaxPolicy.__init__(self, actions)
        self.h = LinearApproximator(state_vector_size + 1,
                                    initialization_cov_constant)


class ACAgent:
    """
    Actor-critic RL agent for maximizing long-run average
    reward over long-run average cost.

    This agent uses linear function approximation for the Q function
    and a softmax policy
    """

    def __init__(self, policy, value_function,
                 policy_lr, v_lr,
                 init_mu_r=0, init_mu_c=0, mu_lr=0.005,
                 mu_floor=0.01,
                 grad_clip_radius=None):
        self.pi = policy
        self.rv = value_function
        self.cv = deepcopy(self.rv)
        self.pi_lr = policy_lr
        self.v_lr = v_lr
        self.mu_r = init_mu_r
        self.mu_c = init_mu_c
        self.mu_lr = mu_lr
        self.mu_floor = mu_floor
        self.grad_clip_radius = grad_clip_radius
        
        self.state = None
        self.action = None
        
        self.reinit_params()

    def reinit_params(self):
        self.pi.reinit_params()
        self.rv.reinit_params()
        self.cv.reinit_params()

    def _mus_floor(self):
        self.mu_r = max(self.mu_r, self.mu_floor)
        self.mu_c = max(self.mu_c, self.mu_floor)
        
    def sample_action(self, state):
        self.state = state
        self.action = self.pi.sample_action(state)
        return self.action
    
    def update(self, reward_cost_tuple, next_state):

        assert self.state is not None, 'sample_action must be called first'
        
        reward, cost = reward_cost_tuple

        self.mu_r = self.mu_lr * reward + (1 - self.mu_lr) * self.mu_r
        self.mu_c = self.mu_lr * cost + (1 - self.mu_lr) * self.mu_c
        self._mus_floor()

        r_td_err = reward - self.mu_r + self.rv(next_state) - self.rv(self.state)
        self.rv.params = self.rv.params \
                + self.v_lr * r_td_err * self.rv.gradient(self.state)

        c_td_err = cost - self.mu_c + self.cv(next_state) - self.cv(self.state)
        self.cv.params = self.cv.params \
                + self.v_lr * c_td_err * self.cv.gradient(self.state)

        self.pi.params = self.pi.params \
                + self.pi_lr * (self.mu_r / self.mu_c) * \
                self.pi.grad_log_policy(self.state, self.action) * \
                (r_td_err / self.mu_r - c_td_err / self.mu_c)


    def change_stepsizes(self, new_actor_stepsize, new_critic_stepsize):
        self.actor_stepsize = new_actor_stepsize
        self.critic_stepsize = new_critic_stepsize
    
    
class LinearACAgent(ACAgent):
    def __init__(self, states, actions,
                 policy_lr, v_lr, init_mu_r=0, init_mu_c=0, mu_lr=0.005,
                 mu_floor=0.01,
                 policy_cov_constant=1, value_func_cov_constant=1,
                 grad_clip_radius=None):
        
        state_vector_len = 1 if isinstance(states[0], numbers.Number) \
                else len(states[0])

        value_func = LinearApproximator(state_vector_len,
                                        value_func_cov_constant)
        
        policy = SoftmaxPolicyLinear(state_vector_len, actions,
                                     policy_cov_constant)
        
        ACAgent.__init__(self, policy, value_func,
                         policy_lr, v_lr,
                         init_mu_r=init_mu_r, init_mu_c=init_mu_c, mu_lr=mu_lr,
                         mu_floor=mu_floor,
                         grad_clip_radius=grad_clip_radius)















# end
