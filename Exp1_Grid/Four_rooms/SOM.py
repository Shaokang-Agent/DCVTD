import numpy as np
import sys
sys.path.append("..")
from ENV.env_four_rooms import EnvGoObstacle
from algorithm.SOM import SOM_learning
import time
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    single = False
    env = EnvGoObstacle(single, -1)
    max_episode = 500
    max_iteration = 10
    max_steps = 100
    Rewards = np.zeros([max_iteration, max_episode])
    for iter in range(max_iteration):
        SOM_learning_i = [SOM_learning(11 * 5, 12, 5) for _ in range(12)]
        visited_num = np.zeros([11, 5])
        for i in range(max_episode):
            state, goal = env.reset()
            count = 0
            done = False
            rewards = np.zeros(12)
            while not done:
                action_list = []
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = [goal[j][0] * 5 + goal[j][1] for j in range(12)]
                    epsilon = np.min([0.9, 0.7+(0.9-0.7)*(i*max_steps+count)/(max_episode*max_steps/3)])
                    action = SOM_learning_i[index].choose_action([s,g], epsilon)
                    action_list.append(action)
                reward, _, next_state = env.step(action_list)
                if count > max_steps:
                    done = True
                else:
                    count += 1
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = [goal[j][0] * 5 + goal[j][1] for j in range(12)]
                    a = action_list[index]
                    r = reward[index]
                    s_ = next_state[index][0] * 5 + next_state[index][1]
                    rewards[index] += r
                    SOM_learning_i[index].learn([s, g], a, r, [s_, g])
                state = next_state
            Rewards[iter][i] = np.mean(rewards)
            if i % 20 == 0:
                print("iteration {}, episode {}".format(iter, i))
    np.save("../PLOT/4_room/4_som_rewards.npy", Rewards)

