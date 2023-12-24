import numpy as np
import sys
sys.path.append("..")
from ENV.env_four_rooms import EnvGoObstacle
from algorithm.Q_learning import Q_learning
import time
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    single = False
    env = EnvGoObstacle(single, -1)
    max_episode = 500
    max_iteration = 10
    max_steps = 100
    steps_mean = np.zeros(max_iteration)
    for iter in range(max_iteration):
        Q_learning_i = [Q_learning(11 * 5, 5) for _ in range(12)]
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
                    epsilon = np.min([0.9, 0.7+(0.9-0.7)*(i*max_steps+count)/(max_episode*max_steps/3)])
                    action = Q_learning_i[index].choose_action([s,g], epsilon)
                    action_list.append(action)
                reward, done, next_state = env.step(action_list)
                if count > max_steps:
                    break
                else:
                    count += 1
                for index in range(12):
                    s = state[index][0] * 5 + state[index][1]
                    g = goal[index][0] * 5 + goal[index][1]
                    a = action_list[index]
                    r = reward[index]
                    s_ = next_state[index][0] * 5 + next_state[index][1]
                    Q_learning_i[index].learn([s,g],a,r,[s_,g])
                state = next_state
            all_steps += count
        print(iter)
        steps_mean[iter] = all_steps / max_episode
    np.save("../PLOT/4_room/4_baseline_steps.npy", steps_mean)

