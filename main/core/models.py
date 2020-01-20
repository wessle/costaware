import torch
import torch.distributions as td
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class PolicyNetwork(nn.Module):
    """Base class for stochastic policy networks."""

    def __init__(self):
        super().__init__()

    def forward(self, state):
        """Take state as input, then output the parameters of the policy."""

        raise NotImplemented("forward not implemented.")

    def sample(self, state):
        """
        Sample an action based on the model parameters given the current state.
        """

        raise NotImplemented("sample not implemented.")


class DirichletPolicyBase(PolicyNetwork):
    """
    Base class for Dirichlet policies.
    
    Desired network needs to be implemented.
    """

    def __init__(self, min_alpha=-np.inf, max_alpha=np.inf):

        super().__init__()

        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def sample(self, state, no_log_prob=False):
        alpha = self.forward(state)
        dist = td.Dirichlet(alpha)
        action = dist.sample()
        if no_log_prob:
            return action
        else:
            return action, dist.log_prob(action)


class DirichletPolicySingleLayer(DirichletPolicyBase):
    """Working, single-layer Dirichlet policy network."""

    def __init__(self, state_dim, action_dim,
                 hidden_layer_size=256,
                 min_alpha=-np.inf, max_alpha=np.inf):

        super().__init__(min_alpha, max_alpha)

        self.linear1 = nn.Linear(state_dim, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, action_dim)

        nn.init.normal_(self.linear1.weight, std=0.001)
        nn.init.normal_(self.linear1.bias, std=0.001)
        nn.init.normal_(self.linear2.weight, std=0.001)
        nn.init.normal_(self.linear2.bias, std=0.001)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        action = self.max_alpha * F.sigmoid(self.linear2(x))
        return torch.clamp(action, self.min_alpha, self.max_alpha)


class DirichletPolicyTwoLayer(DirichletPolicyBase):
    """Working, single-layer Dirichlet policy network."""

    def __init__(self, state_dim, action_dim,
                 hidden_layer1_size=256,
                 hidden_layer2_size=256,
                 min_alpha=-np.inf, max_alpha=np.inf):

        super().__init__(min_alpha, max_alpha)

        self.linear1 = nn.Linear(state_dim, hidden_layer1_size)
        self.linear2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.linear3 = nn.Linear(hidden_layer2_size, action_dim)

        nn.init.normal_(self.linear1.weight, std=0.001)
        nn.init.normal_(self.linear1.bias, std=0.001)
        nn.init.normal_(self.linear2.weight, std=0.001)
        nn.init.normal_(self.linear2.bias, std=0.001)
        nn.init.normal_(self.linear3.weight, std=0.001)
        nn.init.normal_(self.linear3.bias, std=0.001)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = self.max_alpha * F.sigmoid(self.linear3(x))
        return torch.clamp(action,
                           self.min_alpha, self.max_alpha)


class LogitNormalPolicyBase(PolicyNetwork):
    """
    Base class for logistic-normal policies.
    
    This class of policies may have significant advantages over Dirichlet
    policies in more sophisticated settings, as correlation between different
    asset classes can be captured.
    """
