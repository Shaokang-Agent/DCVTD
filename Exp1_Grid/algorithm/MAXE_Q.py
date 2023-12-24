import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class MAXE_Q_learning():
    """docstring for DQN"""
    def __init__(self, state_num, action_num):
        super(MAXE_Q_learning, self).__init__()
        self.training_steps = 0
        self.action_num = action_num
        self.Q_value = np.zeros([state_num, state_num, action_num])
        self.eta = 0.7
        self.tau = 0.5
        self.eps = 1e-5

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def choose_action(self, state, episolon):
        action_value = self.Q_value[state[0], state[1]]
        pi = self.softmax(action_value/0.5)
        log_pi = np.log(pi+self.eps)
        if np.random.rand() <= episolon:
            action = np.random.choice(range(self.action_num), p=pi)
        else:
            action = np.random.randint(0, self.action_num)
        return action, pi, log_pi

    def learn(self, state, action, reward, next_state):
        self.pi = self.softmax(self.Q_value[state[0], state[1]]/0.5)
        self.entropy = np.sum(-self.pi*np.log(self.pi+self.eps))
        self.Q_value[state[0], state[1], action] =  (1 - self.eta) * self.Q_value[state[0], state[1], action] + self.eta * (reward + 0.02 * self.entropy + 0.99 * np.max(self.Q_value[next_state[0], next_state[1]]) )
        self.training_steps += 1
    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q
