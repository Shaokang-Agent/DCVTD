import sys
sys.path.append("..")
import numpy as np
from ENV.env_two_rooms import EnvGoObstacle
from algorithm.EITI_Q import EITI_Q_learning
import math
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    max_episode = 500
    max_iteration = 10
    max_steps = 200
    map_size = 7
    rewards_agent = np.zeros([2, max_iteration, max_episode])
    for num_agent in [5, 10]:
        for iter in tqdm(range(max_iteration)):
            env = EnvGoObstacle(map_size, num_agent)
            Q_learning = [EITI_Q_learning(map_size*map_size, 5) for _ in range(num_agent)]
            visited_num_sa = np.zeros([num_agent, map_size * map_size, 5, num_agent, map_size * map_size, 5, map_size * map_size])
            all_steps = 0
            for i in range(max_episode):
                state,goal = env.reset()
                count = 0
                done = False
                rewards = np.zeros(num_agent)
                while not done:
                    action_list = []
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        g = goal[index][0] * map_size + goal[index][1]
                        epsilon = np.min([0.99, 0.7+(0.99-0.7)*(i*max_steps+count)/(max_episode*max_steps/3)])
                        action = Q_learning[index].choose_action([s,g], epsilon)
                        action_list.append(action)
                    reward, _, next_state = env.step(action_list)
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        g = goal[index][0] * map_size + goal[index][1]
                        a = action_list[index]
                        for other_index in range(num_agent):
                            if other_index != index:
                                s_other = state[other_index][0] * map_size + state[other_index][1]
                                g_other = goal[other_index][0] * map_size + goal[other_index][1]
                                a_other = action_list[other_index]
                                s_next_other = next_state[other_index][0] * map_size + next_state[other_index][1]
                                visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] += 1
                    if count > max_steps:
                        done = True
                    else:
                        count += 1
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        g = goal[index][0] * map_size + goal[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0] * map_size + next_state[index][1]
                        rewards[index] += r
                        r_in = 0
                        for other_index in range(num_agent):
                            if other_index != index:
                                s_other = state[other_index][0] * map_size + state[other_index][1]
                                g_other = goal[other_index][0] * map_size + goal[other_index][1]
                                a_other = action_list[other_index]
                                s_next_other = next_state[other_index][0] * map_size + next_state[other_index][1]
                                if visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] == 0 or np.sum(visited_num_sa[index, s, a, other_index, s_other, a_other, :]) == 0 or np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, :]) == 0:
                                    r_temp = 0
                                else:
                                    r_temp = math.log( (visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] / np.sum(visited_num_sa[index, s, a, other_index, s_other, a_other, :]) ) /
                                                       (np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, s_next_other]) / np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other, :])) )
                                r_in += r_temp
                        Q_learning[index].learn([s,g], a, r, [s_,g], r_in)
                    state = next_state
                rewards_agent[int(num_agent/5-1)][iter][i] = np.mean(rewards)
                all_steps += count
    np.save("../PLOT/2_room/2_eiti_rewards.npy", rewards_agent)
