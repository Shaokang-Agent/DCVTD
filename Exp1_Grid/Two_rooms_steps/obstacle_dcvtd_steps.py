import sys
sys.path.append("..")
import numpy as np
from ENV.env_two_rooms import EnvGoObstacle
from algorithm.DCVTD import DCVTD
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
# Core training parameters
parser.add_argument("--high", type=int, default=7)
parser.add_argument("--width", type=int, default=7)
parser.add_argument("--num_agent", type=int, default=5)

args = parser.parse_args()


if __name__ == '__main__':
    max_episode = 500
    max_iteration = 10
    max_steps = 200
    map_size = 7
    steps_mean = np.zeros([6, max_iteration])
    for num_agent in range(5, 11):
        for iter in tqdm(range(max_iteration)):
            env = EnvGoObstacle(map_size, num_agent)
            DCVTD_Q = [DCVTD(args, map_size, map_size * map_size, 5) for _ in range(num_agent)]
            all_steps = 0
            for i in range(max_episode):
                state, goal = env.reset()
                count = 0
                rewards = np.zeros(num_agent)
                done = False
                while not done:
                    action_list = []
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        g = goal[index][0] * map_size + goal[index][1]
                        epsilon = np.min([0.99, 0.7 + (0.99 - 0.7) * (i * max_steps + count) / (max_episode * max_steps / 3)])
                        action = DCVTD_Q[index].choose_action([s,g], epsilon)
                        action_list.append(action)
                    reward, done, next_state = env.step(action_list)
                    if count > max_steps:
                        break
                    else:
                        count += 1
                    s_total_list, a_total_list, next_s_total_list = [], [], []
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        g = goal[index][0] * map_size + goal[index][1]
                        a = action_list[index]
                        s_ = next_state[index][0] * map_size + next_state[index][1]
                        s_total_list.append([s,g])
                        a_total_list.append(a)
                        next_s_total_list.append([s_,g])
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        g = goal[index][0] * map_size + goal[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0]*map_size + next_state[index][1]
                        s_other_list = s_total_list[:index] + s_total_list[index+1:]
                        a_other_list = a_total_list[:index] + a_total_list[index + 1:]
                        next_s_other_list = next_s_total_list[:index] + next_s_total_list[index + 1:]
                        rewards[index] += r
                        DCVTD_Q[index].learn([s,g], a, r, [s_,g], s_other_list, a_other_list, next_s_other_list)
                    state = next_state
                all_steps += count
            steps_mean[num_agent - 5, iter] = all_steps / max_episode
    np.save("../PLOT/2_room/2_dcvtd_steps.npy", steps_mean)
