import sys
sys.path.append("./")
import numpy as np
import os
from replay_buffer.replay_buffer_episode import ReplayBuffer
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from learners.DQN import DQN
from learners.DCVTD import DCQF
from learners.SOCIAL import DQN_SOCIAL
from learners.MADDPG import MADDPG
from learners.QMIX import QMIX
from tqdm import tqdm

def make_env(args):
    if args.env == "Harvest":
        single_env = HarvestEnv(num_agents=args.num_agents)
        env_name = "HarvestEnv"
        def env_creator(_):
            return HarvestEnv(num_agents=args.num_agents)
    else:
        single_env = CleanupEnv(num_agents=args.num_agents)
        env_name = "CleanupEnv"
        def env_creator(_):
            return CleanupEnv(num_agents=args.num_agents)
    register_env(env_name, env_creator)
    if env_name == "HarvestEnv":
        action_num = 8
    else:
        action_num = 9
    return single_env, action_num


class Runner:
    def __init__(self, args):
        env, action_num = make_env(args)
        self.env = env
        self.args = args
        self.args.action_num = action_num
        if self.args.algorithm == "DQN":
            self.agents = [DQN(args, action_num, agent_id) for agent_id in range(args.num_agents)]
            from run_scripts.rollout import RolloutWorker
        elif self.args.algorithm == "DCVTD":
            self.agents = [DCQF(args, action_num, agent_id) for agent_id in range(self.args.control_num_agents)]
            if self.args.control_num_agents < self.args.num_agents:
                self.agents += [DQN(args, action_num, agent_id) for agent_id in range(self.args.control_num_agents, self.args.num_agents)]
            from run_scripts.rollout import RolloutWorker
        elif self.args.algorithm == "MADDPG":
            self.agents = [MADDPG(args, action_num, agent_id) for agent_id in range(args.num_agents)]
            from run_scripts.rollout import RolloutWorker
        elif self.args.algorithm == "SOCIAL":
            # self.agents = [DQN_SOCIAL(args, action_num, agent_id) for agent_id in range(self.args.control_num_agents)]
            # self.agents += [DQN(args, action_num, agent_id) for agent_id in range(self.args.control_num_agents, self.args.num_agents)]
            self.agents = [DQN_SOCIAL(args, action_num, agent_id) for agent_id in range(self.args.num_agents)]
            from run_scripts.rollout_social import RolloutWorker
        elif self.args.algorithm == "QMIX":
            self.agents = QMIX(args, action_num)
            from run_scripts.rollout_qmix import RolloutWorker
        else:
            self.agents = [DCQF(args, action_num, agent_id) for agent_id in range(args.num_agents)]

        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)

        self.episode_rewards = []
        # self.save_model_path = './model_pkl/' + self.args.env + str(self.args.num_agents) + '/' + self.args.algorithm
        if self.args.algorithm == "DCVTD":
            self.save_data_path = './log/data/' + self.args.env + str(self.args.num_agents) + '/' + self.args.algorithm + "_control_" + str(self.args.control_num_agents)
        else:
            self.save_data_path = './log/data/' + self.args.env + str(self.args.num_agents) + '/' + self.args.algorithm

        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)

    def run(self, num):
        if self.args.algorithm == "DCVTD":
            self.writer = SummaryWriter("./log/runs/" + self.args.env + str(self.args.num_agents) + "/" + self.args.algorithm + "_control_" + str(self.args.control_num_agents) + "/" + str(num))
        else:
            self.writer = SummaryWriter("./log/runs/" + self.args.env + str(self.args.num_agents) + "/" + self.args.algorithm + "/" + str(num))
        train_steps = 0

        for epi in tqdm(range(self.args.num_episodes)):
            print('Run {}, train episode {}'.format(num, epi))
            if epi % self.args.evaluate_cycle == 0:
                episode_individual_reward = self.evaluate()
                episode_reward = np.sum(episode_individual_reward)
                self.episode_rewards.append(episode_individual_reward)
                for i in range(self.args.num_agents):
                    self.writer.add_scalar("Agent_{}_reward".format(str(i)), episode_individual_reward[i], epi)
                self.writer.add_scalar("Total_reward", episode_reward, epi)
                if self.args.algorithm == "DCVTD":
                    print("Environment: {}, agent-number: {}, algorithm: {}, control_number: {}, Episode {}, Episode_reward {}".format(
                        self.args.env, self.args.num_agents, self.args.algorithm, self.args.control_num_agents, epi, episode_reward))
                else:
                    print("Environment: {}, agent-number: {}, algorithm: {}, Episode {}, Episode_reward {}".format(
                        self.args.env, self.args.num_agents, self.args.algorithm, epi, episode_reward))
            episode_data, _ = self.rolloutWorker.generate_episode(epi)
            self.buffer.add(episode_data)
            if self.args.batch_size < self.buffer.__len__():
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.__len__(), self.args.batch_size))
                    if self.args.algorithm == "QMIX":
                        loss = self.agents.learn(mini_batch, num)
                        self.writer.add_scalar("Agent_Total_Loss", loss, train_steps)
                    else:
                        for i in range(self.args.num_agents):
                            if self.args.algorithm == "MADDPG":
                                closs, aloss = self.agents[i].learn(mini_batch, i, self.agents, num)
                                self.writer.add_scalar("Agent_{}_CLoss".format(str(i)), closs, train_steps)
                                self.writer.add_scalar("Agent_{}_ALoss".format(str(i)), aloss, train_steps)
                            else:
                                loss = self.agents[i].learn(mini_batch, i, num)
                                self.writer.add_scalar("Agent_{}_Loss".format(str(i)), loss, train_steps)
                    train_steps += 1
            np.save(self.save_data_path + '/epi_total_reward_{}'.format(str(num)), self.episode_rewards)

    def evaluate(self):
        episode_rewards = 0
        for epi in range(self.args.evaluate_epi):
            _, episode_reward = self.rolloutWorker.generate_episode(epi, evaluate=True)
            episode_rewards += episode_reward
        return episode_rewards / self.args.evaluate_epi
