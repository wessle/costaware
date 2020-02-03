import numpy as np
from random import choice


class TabularQLearner:
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
            self.action = choice(self.actions)
            while self.action == greedy:
                self.action = choice(self.actions)
        else:
            self.action = greedy

        return self.action

    def update(self, reward, cost, next_state):
        """Perform the update step."""

        assert self.state is not None, "sample_action must be called first"

        td_err = reward - self.rho * cost + self.V(next_state) \
                - self.ref_state_val() - self.Q(self.state, self.action)
        self.inplace_add_to_Q_table(self.state, self.action, self.q_lr * td_err)
        self.rho += self.rho_lr * self.ref_state_val()













# end
