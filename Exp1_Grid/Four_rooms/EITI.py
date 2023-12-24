import numpy as np
import sys
sys.path.append("..")
from ENV.env_four_rooms import EnvGoObstacle
from algorithm.EITI_Q import EITI_Q_learning
import math
import time
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    single = False
    env = EnvGoObstacle(single, -1)
    max_episode = 500
    max_iteration = 10
    max_steps = 200
    Rewards = np.zeros([max_iteration, max_episode])
    for iter in range(max_iteration):
        Q_learning = [EITI_Q_learning(11 * 5, 5) for _ in range(12)]
        visited_num_sa = np.zeros([12, 11 * 5, 5, 12, 11 * 5, 5, 11 * 5])
        visited_num = np.zeros([11, 5])
        all_steps = 0
        for i in range(max_episode):
            state,goal = env.reset()
            count = 0
            done = False
            rewards = np.zeros(12)
            while not done:
                action_list = []
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = goal[index][0] * 5 + goal[index][1]
                    epsilon = np.min(
                        [0.99, 0.7 + (0.99 - 0.7) * (i * max_steps + count) / (max_episode * max_steps / 3)])
                    action = Q_learning[index].choose_action([s, g], epsilon)
                    action_list.append(action)
                reward, _, next_state = env.step(action_list)
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = goal[index][0] * 5 + goal[index][1]
                    a = action_list[index]
                    for other_index in range(12):
                        if other_index != index:
                            s_other = state[other_index][0] * 5 + state[other_index][1]
                            g_other = goal[other_index][0] * 5 + goal[other_index][1]
                            a_other = action_list[other_index]
                            s_next_other = next_state[other_index][0] * 5 + next_state[other_index][1]
                            visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] += 1
                if count > max_steps:
                    done = True
                else:
                    count += 1
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = goal[index][0] * 5 + goal[index][1]
                    a = action_list[index]
                    r = reward[index]
                    s_ = next_state[index][0] * 5 + next_state[index][1]
                    rewards[index] += r
                    r_in = 0
                    for other_index in range(12):
                        if other_index != index:
                            s_other = state[other_index][0] * 5 + state[other_index][1]
                            g_other = goal[other_index][0] * 5 + goal[other_index][1]
                            a_other = action_list[other_index]
                            s_next_other = next_state[other_index][0] * 5 + next_state[other_index][1]
                            if visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] == 0 or np.sum(visited_num_sa[index, s, a, other_index, s_other, a_other, :]) == 0 or np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, :]) == 0:
                                r_temp = 0
                            else:
                                r_temp = math.log( (visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] / np.sum(visited_num_sa[index, s, a, other_index, s_other, a_other, :]) ) /
                                                   (np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, s_next_other]) / np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, :])) )
                            r_in += r_temp
                    Q_learning[index].learn([s,g], a, r, [s_,g], r_in)
                state = next_state
            Rewards[iter][i] = np.mean(rewards)
            if i == max_episode-1:
                print("iteration {}, state {}, goal {}".format(iter, state, goal))
    np.save("../PLOT/4_room/4_eiti_rewards.npy", Rewards)

