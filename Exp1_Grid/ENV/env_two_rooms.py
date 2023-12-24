import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from collections import Counter
import cv2
import copy

class EnvGoObstacle(object):
    def __init__(self, map_size, num_agent):
        self.map_size = map_size
        self.num_agent = num_agent
        self.state = []
        self.goal = []
        self.occupancy = np.zeros((self.map_size, self.map_size))
        self.obstacles = np.zeros((self.map_size, self.map_size))
        inter_length = int(self.map_size / 2)
        for j in range(self.map_size):
            if j != inter_length:
                self.obstacles[inter_length, j] = 1
        goal = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        for i in range(self.num_agent):
            while self.obstacles[goal[0]][goal[1]] == 1 or goal in self.goal:
                goal = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            self.goal.append(goal)

    def reset(self):
        self.state = []
        self.goal = []
        goal = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        for i in range(self.num_agent):
            while self.obstacles[goal[0]][goal[1]] == 1 or goal in self.goal:
                goal = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            self.goal.append(goal)
        for i in range(self.num_agent):
            state = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            while self.obstacles[state[0]][state[1]] == 1 or state in self.state:
                state = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            self.state.append(state)
        return self.state, self.goal

    def get_env_info(self):
        return 0

    def get_reward(self, state, action_list):
        reward = np.zeros(self.num_agent) - 0.1
        next_state = copy.deepcopy(state)
        for i in range(self.num_agent):
            if action_list[i] == 0:  # move right
                next_state[i][1] = state[i][1] + 1
            elif action_list[i] == 1:  # move left
                next_state[i][1] = state[i][1] - 1
            elif action_list[i] == 2:  # move up
                next_state[i][0] = state[i][0] - 1
            elif action_list[i] == 3:  # move down
                next_state[i][0] = state[i][0] + 1
            elif action_list[i] == 4:  # stay
                pass
        for i in range(self.num_agent):
            other_next_state = next_state[:i] + next_state[i+1:]
            other_state = state[:i] + state[i+1:]
            if next_state[i] == self.goal[i]:
                reward[i] = 1
            elif next_state[i][0] < 0 or next_state[i][0] > self.map_size-1 or next_state[i][1] < 0 or next_state[i][1] > self.map_size-1 or self.obstacles[next_state[i][0]][next_state[i][1]] == 1:
                next_state[i] = state[i]
                reward[i] = -0.3
            elif next_state[i] in other_next_state:
                next_state[i] = state[i]
                reward[i] = -1
            elif next_state[i] in other_state:
                for j in range(len(other_state)):
                    if next_state[i] == other_state[j] and state[i] == other_next_state[j]:
                        next_state[i] = state[i]
                        reward[i] = -1
        return reward, next_state

    def step(self, action_list):
        reward, next_state = self.get_reward(self.state, action_list)
        done = True
        for i in range(self.num_agent):
            if next_state[i] != self.goal[i]:
                done = False
        self.state = next_state
        return reward, done, self.state

    def sqr_dist(self, pos1, pos2):
        return (pos1[0]-pos2[0])*(pos1[0]-pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1])
