import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class SOM_learning():
    """docstring for DQN"""
    def __init__(self, state_num, num_agent, action_num):
        super(SOM_learning, self).__init__()
        self.eta = 0.7
        self.state_num = state_num
        self.action_num = action_num
        self.num_agent = num_agent
        self.input = [self.state_num]
        for i in range(self.num_agent):
            self.input.append(self.state_num)
        self.input.append(action_num)
        self.Q_value = np.zeros(self.input)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def choose_action(self, state, episolon):
        s = np.array(state[0])
        g = np.array(state[1])
        inputs = np.insert(g, 0, s)
        if np.random.rand() <= episolon:
            action_value = self.Q_value[inputs]
            action_value_prob = self.softmax(action_value/0.5)
            action = np.random.choice(range(self.action_num), p=action_value_prob)
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def learn(self, state, action, reward, next_state):
        s = np.array(state[0])
        g = np.array(state[1])
        ns = np.array(next_state[0])
        ng = np.array(next_state[1])
        inputs = np.insert(g, 0, s)
        ninputs = np.insert(ng, 0, ns)
        self.Q_value[inputs, action] = (1 - self.eta) * self.Q_value[inputs, action] +  self.eta * (reward + 0.99 * np.max(self.Q_value[ninputs]))

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q
