import numpy as np
import torch
import warnings

import main.utils.utils as utils


class RLAgent:
    """Base class for agents corresponding to specific RL algorithms.
    """

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


class RVIQLearningBasedAgent(RLAgent):
    """
    RVI Q-learning-based RL agent for maximizing long-run average
    reward over long-run average cost.

    Note that the set of actions must be finite.
    """

    # TODO: which attributes and methods should be private?

    def __init__(self, buffer_maxlen, batchsize, actions,
                 q_network, q_lr, rho_lr,
                 eps=0.01, enable_cuda=True, optimizer=torch.optim.Adam,
                 grad_clip_radius=None, rho_init=0.0, rho_clip_radius=None):

        self.buffer = utils.Buffer(buffer_maxlen)
        self.N = batchsize
        self.actions = actions # numpy array of actions
        self.q = q_network
        self.q_loss = torch.nn.MSELoss()

        self.eps = eps
        self.__enable_cuda = enable_cuda
        self.enable_cuda(self.__enable_cuda, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called
        self.torch_actions = list(utils.arrays_to_tensors(actions, self.device))

        self.q_optim = optimizer(self.q.parameters(), lr=q_lr)
        self.rho_lr = rho_lr
        self.grad_clip_radius = grad_clip_radius
        self.rho = rho_init
        self.rho_clip_radius = rho_clip_radius

        self.state = None
        self.action = None

    def action_values(self, state):
        """Return list of action values for current state."""

        with torch.no_grad():
            state = utils.array_to_tensor(state, self.device)
            values = [self.q(torch.cat([state, action])).item()
                      for action in self.torch_actions]

        return values

    def state_value(self, state):
        """Return value estimate of the current state."""

        return max(self.action_values(state))

    def state_values(self, states):
        """Return value estimates of an array of states."""

        return [self.state_value(state) for state in states]

    def sample_action(self, state):
        """
        Sample an action epsilon-greedily.
        
        An action must be sampled before self.update can be called.
        """

        self.state = state
        if np.random.uniform() < self.eps:
            self.action = np.random.choice(self.actions)
        else:
            self.action = self.actions[np.argmax(self.action_values(state))]

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

        new_sample = (self.state, self.action, reward_cost_tuple, next_state)
        self.buffer.add(new_sample)

        if len(self.buffer) >= self.N:
            states, actions, rewards, costs, next_states = \
                    utils.arrays_to_tensors(self.buffer.sample_batch(self.N),
                                            self.device)

            # assemble pieces for the Q update})
            with torch.no_grad():
                proxy_rewards = rewards - self.rho * costs
                average_reward = torch.mean(proxy_rewards) * torch.ones(self.N)
                state_values = utils.array_to_tensor(self.state_values(states),
                                                     self.device)
                q_targets = proxy_rewards - average_reward + state_values

            # form the loss function and take a gradient step
            q_inputs = torch.cat([states, actions], dim=1)
            q_estimates = self.q(q_inputs)
            loss = self.q_loss(q_targets.unsqueeze(dim=1), q_estimates)
            self.q_optim.zero_grad()
            loss.backward()
            if self.grad_clip_radius is not None:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(),
                                               self.grad_clip_radius)
            self.q_optim.step()

            # perform the rho update
            rho_clip_radius = np.inf if self.rho_clip_radius is None \
                    else self.rho_clip_radius
            self.rho += min(rho_clip_radius,
                            self.rho_lr * np.average(state_values))

    def save_models(self, filename):
        """Save Q function and optimizer."""

        torch.save({
                'using_cuda': self.__enable_cuda,
                'q_state_dict': self.q.state_dict(),
                'q_optim_state_dict': self.q_optim.state_dict(),
        }, filename)

    def load_models(self, filename, continue_training=True):
        """Load Q function and optimizer."""
        
        model = torch.load(filename)

        self.__enable_cuda = model['using_cuda']
        self.q.load_state_dict(model['q_state_dict'])
        self.q_optim.load_state_dict(model['q_optim_state_dict'])
        
        self.q.train() if continue_training \
            else self.q.eval()

        self.enable_cuda(self.__enable_cuda, warn=False)
        




# end
