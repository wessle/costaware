import numpy as np
import torch
import torch.nn.functional as F
import copy
import warnings
import wesutils
import numbers
from scipy.special import softmax
from copy import deepcopy


# RL agents using batch learning and neural nets for function approximators

class DeepRLAgent:
    """Base class for agents corresponding to specific RL algorithms."""

    def __init__(self):
        raise NotImplemented("__init__ not implemented.")

    def sample_action(self, state):
        raise NotImplemented("sample_action not implemented.")

    def enable_cuda(self, enable_cuda=True, warn=True):
        """
        Enable or disable CUDA. Issue warning that converting after
        initializing optimizers can cause undefined behavior when using
        optimizers other than Adam or SGD.
        """

        raise NotImplemented("enable_cuda not implemented.")

    def update(self, reward_cost_tuple, next_state):
        raise NotImplemented("update not implemented.")

    def save_models(self):
        raise NotImplemented("save_models not implemented.")

    def load_models(self):
        raise NotImplemented("load_models not implemented.")


class DeepRVIQLearningBasedAgent(DeepRLAgent):
    """
    RVI Q-learning-based RL agent for maximizing long-run average
    reward over long-run average cost.

    Note that the set of actions must be finite.
    """

    def __init__(self, buffer_maxlen, batchsize, actions,
                 q_network,
                 q_lr, rho_lr,
                 eps=0.1, enable_cuda=True, optimizer=torch.optim.Adam,
                 grad_clip_radius=None,
                 rho_init=0.0, rho_clip_radius=None):

        self.buffer = wesutils.Buffer(maxlen=buffer_maxlen)
        self.N = batchsize
        self.actions = actions # numpy array of actions
        self.q = q_network
        self.q_loss = torch.nn.MSELoss()

        self.eps = eps
        self.__enable_cuda = enable_cuda
        self.enable_cuda(self.__enable_cuda, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called
        self.torch_actions = torch.FloatTensor(actions).to(
            self.device).unsqueeze(dim=1)

        self.q_optim = optimizer(self.q.parameters(), lr=q_lr)
        self.rho_lr = rho_lr
        self.grad_clip_radius = grad_clip_radius
        self.rho = rho_init
        self.rho_clip_radius = np.inf if rho_clip_radius is None \
            else rho_clip_radius

        self.state = None
        self.action = None
        self.ref_state = None

        self.ref_val_est = 0

    def set_reference_state(self, state):
        """Set the reference state to be used in updates."""

        self.ref_state = wesutils.array_to_tensor(state, self.device)

    def action_values(self, state):
        """
        Return list of action values for current state.
        State must be a torch tensor.
        """

        with torch.no_grad():
            values = self.q(torch.cat([
                state.repeat(self.torch_actions.size()[0], 1),
                self.torch_actions], axis=1))
        return values

    def state_value(self, state):
        """Return value estimate of the current state."""

        return torch.max(self.action_values(state))

    def state_values(self, states):
        """Return value estimates of an array of states."""

        return torch.FloatTensor(
            [self.state_value(state) for state in states]).to(self.device)

    def ref_state_val(self):

        return self.state_value(self.ref_state)

    def sample_action(self, state):
        """
        Sample an action epsilon-greedily.
        "
        An action must be sampled before self.update can be called.
        """

        self.state = state
        state = wesutils.array_to_tensor(state, self.device)
        greedy_index = torch.argmax(self.action_values(state)).item()
        if np.random.uniform() < self.eps:
            random_index = np.random.randint(len(self.actions))
            while random_index == greedy_index:
                random_index = np.random.randint(len(self.actions))
            self.action = self.actions[random_index]
        else:
            self.action = self.actions[greedy_index]

        return self.action

    def enable_cuda(self, enable_cuda=True, warn=True):
        """Enable or disable cuda and update model."""
        
        if warn:
            warnings.warn("Converting models between 'cpu' and 'cuda' after "
                          "initializing optimizers can give errors when using "
                          "optimizers other than SGD or Adam!")
        
        self.__enable_cuda = enable_cuda
        self.device = torch.device(
                'cuda' if torch.cuda.is_available() and self.__enable_cuda \
                else 'cpu')
        self.q.to(self.device)

    def update(self, reward_cost_tuple, next_state):
        """Perform the update step."""

        assert self.state is not None, "sample_action must be called first"

        reward, cost = reward_cost_tuple
        new_sample = (self.state, self.action, reward, cost, next_state)
        self.buffer.append(new_sample)

        if len(self.buffer) >= self.N:
            states, actions, rewards, costs, next_states = \
                    wesutils.arrays_to_tensors(self.buffer.sample(self.N),
                                            self.device)

            # assemble pieces for the Q update
            proxy_rewards = rewards - self.rho * costs
            average_reward = self.ref_state_val() * torch.ones(
                self.N, device=self.device)
            next_state_values = self.state_values(next_states)
            q_targets = proxy_rewards - average_reward + next_state_values

            # form the loss function and take a gradient step
            q_inputs = torch.cat([states, actions.unsqueeze(dim=1)], dim=1)
            q_estimates = self.q(q_inputs)

            loss = self.q_loss(q_targets.unsqueeze(dim=1), q_estimates)
            self.q_optim.zero_grad()
            loss.backward()
            if self.grad_clip_radius is not None:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(),
                                               self.grad_clip_radius)
            self.q_optim.step()

            # perform the rho update
            # ref_state_val = self.ref_state_val().item()
            self.ref_val_est = 0.99 * self.ref_val_est + 0.01 * proxy_rewards.mean()
            ref_state_val = self.ref_val_est
            self.rho += np.sign(ref_state_val) * min(
                self.rho_clip_radius, self.rho_lr * abs(ref_state_val))

    def save_models(self, filename):
        """Save Q function, optimizer, rho estimate."""

        torch.save({
                'using_cuda': self.__enable_cuda,
                'q_state_dict': self.q.state_dict(),
                'q_optim_state_dict': self.q_optim.state_dict(),
                'rho': self.rho,
        }, filename)

    def load_models(self, filename, continue_training=True):
        """Load Q function, optimizer, rho estimate."""
        
        model = torch.load(filename)

        self.__enable_cuda = model['using_cuda']
        self.q.load_state_dict(model['q_state_dict'])
        self.q_optim.load_state_dict(model['q_optim_state_dict'])
        self.rho = model['rho']
        
        self.q.train() if continue_training \
            else self.q.eval()

        self.enable_cuda(self.__enable_cuda, warn=False)
        

class DeepACAgent(DeepRLAgent):
    """
    Actor-critic RL agent for maximizing long-run average
    reward over long-run average cost.
    """

    def __init__(self, buffer_maxlen, batchsize,
                 policy_network, v_network,
                 policy_lr, v_lr,
                 init_mu_r=0, init_mu_c=0, mu_lr=0.005,
                 enable_cuda=True,
                 policy_optimizer=torch.optim.Adam,
                 v_optimizer=torch.optim.Adam,
                 grad_clip_radius=None,
                 reward_cost_mean_floor=1e-8):

        self.buffer = wesutils.Buffer(buffer_maxlen)
        self.N = batchsize
        self.pi = policy_network
        self.pi_optim = policy_optimizer(self.pi.parameters(), lr=policy_lr)
        self.rv = v_network
        self.rv_optim = v_optimizer(self.rv.parameters(), lr=v_lr)
        self.rv_loss = torch.nn.MSELoss()
        self.cv = copy.deepcopy(self.rv)
        self.cv_optim = v_optimizer(self.cv.parameters(), lr=v_lr)
        self.cv_loss = torch.nn.MSELoss()
        self.grad_clip_radius = grad_clip_radius
        self.mu_r = init_mu_r
        self.mu_c = init_mu_c
        self.mu_lr = mu_lr

        self.__enable_cuda = enable_cuda
        self.enable_cuda(self.__enable_cuda, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called

        self.reward_cost_mean_floor = reward_cost_mean_floor

        self.state = None
        self.action = None

    def sample_action(self, state):
        """
        Sample an action using the current policy.
        
        An action must be sampled before self.update can be called.
        """

        self.state = state
        state = wesutils.array_to_tensor(state, self.device)
        self.action = self.pi.sample(
            state, no_log_prob=True).cpu().detach().numpy()
        return self.action

    def enable_cuda(self, enable_cuda=True, warn=True):
        """Enable or disable cuda and update models."""
        
        if warn:
            warnings.warn("Converting models between 'cpu' and 'cuda' after "
                          "initializing optimizers can give errors when using "
                          "optimizers other than SGD or Adam!")
        
        self.__enable_cuda = enable_cuda
        self.device = torch.device(
                'cuda' if torch.cuda.is_available() and self.__enable_cuda \
                else 'cpu')
        self.pi.to(self.device)
        self.rv.to(self.device)
        self.cv.to(self.device)
        
    def _mean_floor(self, val):
        return torch.clamp(val, self.reward_cost_mean_floor, np.inf)

    def away_from_zero(self, val):
        return torch.sign(val) * max(self.reward_cost_mean_floor, torch.abs(val))

    def update(self, reward_cost_tuple, next_state):
        """Perform the update step."""

        assert self.state is not None, "sample_action must be called first"

        reward, cost = reward_cost_tuple
        new_sample = (self.state, self.action, reward, cost, next_state)
        self.buffer.append(new_sample)

        # NOTE: the mu values below are not being used at the moment,
        # but I'm keeping them around just in case
        self.mu_r = self.mu_lr * reward + (1 - self.mu_lr) * self.mu_r
        self.mu_c = self.mu_lr * cost + (1 - self.mu_lr) * self.mu_c

        if len(self.buffer) >= self.N:
            # actions are not needed for these updates, so ignore them
            states, _, rewards, costs, next_states = \
                    wesutils.arrays_to_tensors(self.buffer.sample(self.N),
                                            self.device)

            rewards = rewards.unsqueeze(dim=1)
            costs = costs.unsqueeze(dim=1)

            # assemble pieces needed for policy and value function updates
            # TODO: make better use of torch.no_grad() to improve memory efficiency
            r_mean = self.away_from_zero(self.rv(states).mean())
            r_next_state_vals = self.rv(next_states)
            r_targets = rewards - r_mean*torch.ones(self.N, 1, device=self.device) \
                    + r_next_state_vals
            r_state_vals = self.rv(states)

            c_mean = self.away_from_zero(self.cv(states).mean())
            c_next_state_vals = self.cv(next_states)
            c_targets = costs - c_mean*torch.ones(self.N, 1, device=self.device) \
                    + c_next_state_vals
            c_state_vals = self.cv(states)

            # value updates
            r_loss = self.rv_loss(r_targets.detach(), r_state_vals)
            self.rv_optim.zero_grad()
            r_loss.backward()
            if self.grad_clip_radius is not None:
                torch.nn.utils.clip_grad_norm_(self.rv.parameters(),
                                               self.grad_clip_radius)
            self.rv_optim.step()

            c_loss = self.cv_loss(c_targets.detach(), c_state_vals)
            self.cv_optim.zero_grad()
            c_loss.backward()
            if self.grad_clip_radius is not None:
                torch.nn.utils.clip_grad_norm_(self.rv.parameters(),
                                               self.grad_clip_radius)
            self.cv_optim.step()

            # policy update
            with torch.no_grad():
                r_td_err = (r_targets - r_state_vals)
                c_td_err = (c_targets - c_state_vals)
                err_vector = ((r_mean/c_mean)*(
                    r_td_err/r_mean - c_td_err/c_mean)).squeeze()
            _, log_pis = self.pi.sample(states)

#            import pdb; pdb.set_trace()

            pi_loss = -err_vector.dot(log_pis)
            self.pi_optim.zero_grad()
            pi_loss.backward()
            if self.grad_clip_radius is not None:
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                               self.grad_clip_radius)
            self.pi_optim.step()


    def save_models(self, filename):
        """Save models and optimizers."""

        torch.save({
                'using_cuda': self.__enable_cuda,
                'pi_state_dict': self.pi.state_dict(),
                'pi_optim_state_dict': self.pi_optim.state_dict(),
                'rv_state_dict': self.rv.state_dict(),
                'rv_optim_state_dict': self.rv_optim.state_dict(),
                'cv_state_dict': self.cv.state_dict(),
                'cv_optim_state_dict': self.cv_optim.state_dict(),
        }, filename)

    def load_models(self, filename, continue_training=True):
        """Load models and optimizers."""
        
        models = torch.load(filename)

        self.__enable_cuda = models['using_cuda']
        self.pi.load_state_dict(models['pi_state_dict'])
        self.pi_optim.load_state_dict(models['pi_optim_state_dict'])
        self.rv.load_state_dict(models['rv_state_dict'])
        self.rv_optim.load_state_dict(models['rv_optim_state_dict'])
        self.cv.load_state_dict(models['cv_state_dict'])
        self.cv_optim.load_state_dict(models['cv_optim_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.rv.train()
            self.cv.train()
        else:
            self.pi.eval()
            self.rv.eval()
            self.cv.eval()

        self.enable_cuda(self.__enable_cuda, warn=False)


# Simpler agents for continuing MDPs

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


class ContinuingACAgent:
    """
    Actor-critic RL agent for maximizing long-run average
    reward over long-run average cost in the continuing MDP setting.

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
    
    
class LinearACAgent(ContinuingACAgent):
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
        
        ContinuingACAgent.__init__(self, policy, value_func,
                         policy_lr, v_lr,
                         init_mu_r=init_mu_r, init_mu_c=init_mu_c, mu_lr=mu_lr,
                         mu_floor=mu_floor,
                         grad_clip_radius=grad_clip_radius)
