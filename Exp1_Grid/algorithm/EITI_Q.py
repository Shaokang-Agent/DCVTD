import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class EITI_Q_learning():
    """docstring for DQN"""
    def __init__(self, state_num, action_num):
        super(EITI_Q_learning, self).__init__()
        self.training_steps = 0
        self.eta = 0.7
        self.action_num = action_num
        self.Q_value = np.zeros([state_num, state_num, action_num])

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def choose_action(self, state, episolon):
        if np.random.rand() <= episolon:
            action_value = self.Q_value[state[0], state[1]]
            action_value_prob = self.softmax(action_value/0.5)
            action = np.random.choice(range(self.action_num), p=action_value_prob)
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def learn(self, state, action, reward, next_state, intrisic_reward):
        intrisic_reward_shaping = np.clip(0.05*intrisic_reward, a_min=-0.5, a_max=0.5)
        if self.training_steps % 20000 == 0:
            print(intrisic_reward_shaping)
        self.Q_value[state[0], state[1],action] = (1 - self.eta) * self.Q_value[state[0], state[1], action] + self.eta * ( reward + intrisic_reward_shaping + 0.99 * np.max(self.Q_value[next_state[0], next_state[1]]) )
        self.training_steps += 1

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q
