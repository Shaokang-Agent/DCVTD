import numpy as np
import sys
sys.path.append("..")
from ENV.env_four_rooms import EnvGoObstacle
from algorithm.DCV import DCV
import time
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
# Core training parameters
parser.add_argument("--high", type=int, default=11)
parser.add_argument("--width", type=int, default=5)
parser.add_argument("--num_agent", type=int, default=12)

args = parser.parse_args()

if __name__ == '__main__':
    single = False
    env = EnvGoObstacle(single, -1)
    max_episode = 500
    max_iteration = 10
    max_steps = 100
    steps_mean = np.zeros(max_iteration)
    for iter in range(max_iteration):
        DCV_Q = [DCV(args, 5, 11 * 5, 5) for _ in range(12)]
        visited_num = np.zeros([11, 5])
        all_steps = 0
        for i in range(max_episode):
            state, goal = env.reset()
            count = 0
            done = False
            while not done:
                action_list = []
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = goal[index][0] * 5 + goal[index][1]
                    epsilon = np.min(
                        [0.9, 0.7 + (0.9 - 0.7) * (i * max_steps + count) / (max_episode * max_steps / 3)])
                    action = DCV_Q[index].choose_action([s, g], epsilon)
                    action_list.append(action)
                reward, done, next_state = env.step(action_list)
                if count > max_steps:
                    break
                else:
                    count += 1
                s_total_list, a_total_list, next_s_total_list = [], [], []
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = goal[index][0] * 5 + goal[index][1]
                    a = action_list[index]
                    s_ = next_state[index][0] * 5 + next_state[index][1]
                    s_total_list.append([s, g])
                    a_total_list.append(a)
                    next_s_total_list.append([s_, g])
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = goal[index][0] * 5 + goal[index][1]
                    a = action_list[index]
                    r = reward[index]
                    s_ = next_state[index][0] * 5 + next_state[index][1]
                    s_other_list = s_total_list[:index] + s_total_list[index + 1:]
                    a_other_list = a_total_list[:index] + a_total_list[index + 1:]
                    next_s_other_list = next_s_total_list[:index] + next_s_total_list[index + 1:]
                    DCV_Q[index].learn([s, g], a, r, [s_, g], s_other_list, a_other_list, next_s_other_list)
                state = next_state
            all_steps += count
        print(iter)
        steps_mean[iter] = all_steps / max_episode
    np.save("../PLOT/4_room/4_dcv_steps.npy", steps_mean)
