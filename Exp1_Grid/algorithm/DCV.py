import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class DCV():
    """docstring for DQN"""
    def __init__(self, args, state_num, action_num):
        super(DCV, self).__init__()
        self.args = args
        self.training_steps = 0
        self.eta = 0.7
        self.intrinsic_alpha = 0.3
        self.intrinsic_coefficient = 0.04
        self.intrinsic_clip = 0.1
        self.temperature = 0.5
        self.intrinsic_steps = 10000
        if state_num == 49:
            self.intrinsic_coefficient = 0.1
            self.intrinsic_clip = 0.25
            self.temperature = 0.05
            self.intrinsic_steps = 10000
            self.intrinsic_alpha = 0.3
        self.action_num = action_num
        self.Q_value = np.zeros([state_num, state_num, action_num])
        self.DCQ_value = np.zeros([state_num, state_num, state_num, state_num, action_num, action_num])

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def choose_action(self, state, episolon):
        if np.random.rand() <= episolon:
            action_value = self.Q_value[state[0], state[1]]
            action_value_prob = self.softmax(action_value/self.temperature)
            action = np.random.choice(range(self.action_num), p=action_value_prob)
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def learn(self, s, a, r, s_, s_other_list, a_other_list, next_s_other_list):
        self.intrinsic_alpha = np.max([0.01, self.intrinsic_alpha-0.29/self.intrinsic_steps])
        internal_reward = 0
        for i in range(self.args.num_agent-1):
            other_action_value = self.Q_value[next_s_other_list[i][0], next_s_other_list[i][1]]
            other_action_value_prob = self.softmax(other_action_value/self.temperature)
            other_next_action = np.random.choice(range(self.action_num), p=other_action_value_prob)

            self.DCQ_value[s[0], s[1], s_other_list[i][0], s_other_list[i][1], a, a_other_list[i]] = (1 - self.eta) * self.DCQ_value[s[0], s[1], s_other_list[i][0], s_other_list[i][1], a, a_other_list[i]]  + self.eta * ( r + 0.99 * np.max(
                self.DCQ_value[s_[0], s_[1], next_s_other_list[i][0], next_s_other_list[i][1], :, other_next_action]) )
            internal_reward += np.max(self.Q_value[s_other_list[i][0], s_other_list[i][1]]) - np.max(self.DCQ_value[
                s_other_list[i][0], s_other_list[i][1], s[0], s[1], :, a])

        internal_reward = np.clip(self.intrinsic_coefficient*internal_reward/self.args.num_agent, a_min=-self.intrinsic_clip, a_max=self.intrinsic_clip)
        if self.training_steps % 20000 == 0:
            print(internal_reward)
        self.Q_value[s[0], s[1], a] = (1 - self.eta) * self.Q_value[s[0], s[1], a] + self.eta * (
                r + self.intrinsic_alpha * internal_reward + 0.99 * np.max(self.Q_value[s_[0], s_[1]]))

        self.training_steps += 1

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q
