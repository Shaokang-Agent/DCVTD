import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
from collections import Counter
import cv2
import copy

class EnvGoObstacle(object):
    def __init__(self, single=False, agent_index=-1):
        self.high = 11
        self.width = 5
        self.num_agent = 12 if agent_index == -1 else 1
        self.state = []
        self.goal = []
        self.obstacles = np.zeros((self.high, self.width))
        self.occupancy = np.zeros((self.high, self.width))
        self.single = single
        self.agent_index = agent_index

    def reset(self):
        self.state = []
        self.goal = []

        self.obstacles[0, 2] = 1
        self.obstacles[2, 2] = 1
        self.obstacles[4, 2] = 1
        self.obstacles[6, 2] = 1
        self.obstacles[8, 2] = 1
        self.obstacles[10, 2] = 1

        self.obstacles[3, 2] = 1
        self.obstacles[3, 3] = 1
        self.obstacles[3, 4] = 1
        self.obstacles[7, 2] = 1
        self.obstacles[7, 3] = 1
        self.obstacles[7, 4] = 1


        if self.single:
            goal = [[1,4], [1,4], [5,4], [5,4], [9,4], [9,4], [5,0], [5,0], [5,0], [5,0], [5,0], [5,0]]
            self.goal.append(goal[self.agent_index])
            state = [random.randint(0, self.high-1), random.randint(0, self.width-1)]
            while state in self.goal or self.obstacles[state[0]][state[1]] == 1:
                state = [random.randint(0, self.high-1), random.randint(0, self.width-1)]
            self.state.append(state)

        else:
            # self.goal = [[1, 4], [1, 4], [5, 4], [5, 4], [9, 4], [9, 4], [5, 0], [5, 0], [5, 0], [5, 0], [5, 0], [5, 0]]
            # self.state = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0], [10, 0], [0, 4], [2, 4], [4, 4], [6, 4], [8, 4], [10, 4]]
            goal = [random.randint(0, self.high - 1), random.randint(0, self.width - 1)]
            for i in range(self.num_agent):
                while self.obstacles[goal[0]][goal[1]] == 1 or goal in self.goal:
                    goal = [random.randint(0, self.high - 1), random.randint(0, self.width - 1)]
                self.goal.append(goal)
            for i in range(self.num_agent):
                state = [random.randint(0, self.high - 1), random.randint(0, self.width - 1)]
                while self.obstacles[state[0]][state[1]] == 1 or state in self.state:
                    state = [random.randint(0, self.high - 1), random.randint(0, self.width - 1)]
                self.state.append(state)
        return self.state, self.goal

    def get_env_info(self):
        return 0

    def get_reward(self, state, action_list):
        reward = np.zeros(self.num_agent) - 0.01
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
            elif next_state[i][0] < 0 or next_state[i][0] > self.high-1 or next_state[i][1] < 0 or next_state[i][1] > self.width-1 or self.obstacles[next_state[i][0]][next_state[i][1]] == 1:
                next_state[i] = state[i]
                reward[i] = -1
            elif next_state[i] in other_next_state:
                for j in range(len(other_next_state)):
                    if next_state[i] == other_next_state[j] and other_next_state[j] != self.goal[j]:
                        next_state[i] = state[i]
                        reward[i] = -0.5
                    elif next_state[i] == other_next_state[j] and other_next_state[j] == self.goal[j]:
                        reward[i] = -0.01
                    else:
                        pass
            elif next_state[i] in other_state:
                next_state[i] = state[i]
                reward[i] = -0.5
        return reward, next_state

    def step(self, action_list):
        reward, next_state = self.get_reward(self.state, action_list)
        done = True
        for i in range(self.num_agent):
            if next_state[i] != self.goal[i]:
                done = False
        self.state = copy.deepcopy(next_state)
        return reward, done, self.state

    def sqr_dist(self, pos1, pos2):
        return (pos1[0]-pos2[0])*(pos1[0]-pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1])
