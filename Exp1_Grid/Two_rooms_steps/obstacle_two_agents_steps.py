import sys
sys.path.append("..")
import numpy as np
from ENV.env_two_rooms import EnvGoObstacle
from algorithm.Q_learning import Q_learning
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    max_episode = 500
    max_iteration = 10
    max_steps = 200
    map_size = 7
    steps_mean = np.zeros([6, max_iteration])
    rewards_agent = np.zeros([2, max_iteration, max_episode])
    for num_agent in range(5, 11):
        for iter in tqdm(range(max_iteration)):
            env = EnvGoObstacle(map_size, num_agent)
            Q_learning_i = [Q_learning(map_size*map_size, 5) for _ in range(num_agent)]
            all_steps = 0
            for i in range(max_episode):
                state, goal = env.reset()
                count = 0
                done = False
                rewards = np.zeros(num_agent)
                while not done:
                    action_list = []
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        g = goal[index][0] * map_size + goal[index][1]
                        epsilon = np.min([0.99, 0.7+(0.99-0.7)*(i*max_steps+count)/(max_episode*max_steps/3)])
                        action = Q_learning_i[index].choose_action([s,g], epsilon)
                        action_list.append(action)
                    reward, done, next_state = env.step(action_list)
                    if count > max_steps:
                        break
                    else:
                        count += 1
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        g = goal[index][0] * map_size + goal[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0] * map_size + next_state[index][1]
                        rewards[index] += r
                        Q_learning_i[index].learn([s,g],a,r,[s_,g])
                    state = next_state
                all_steps += count
            steps_mean[num_agent - 5, iter] = all_steps / max_episode
    np.save("../PLOT/2_room/2_baseline_steps.npy", steps_mean)
