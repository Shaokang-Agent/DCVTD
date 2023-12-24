import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import os

class QNet(nn.Module):
    """docstring for Net"""
    def __init__(self, action_num, args):
        super(QNet, self).__init__()
        self.args = args
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(169*6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        Batch, seq_len = x.shape[0], x.shape[1]
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = F.leaky_relu(self.cnn(x))
        x = x.reshape(Batch, seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x

class DQN():
    """docstring for DQN"""
    def __init__(self, args, action_num, agent_id):
        super(DQN, self).__init__()
        self.agent_id = agent_id
        self.args = args
        self.eval_net, self.target_net = QNet(action_num, args).float(), QNet(action_num, args).float()
        if self.args.algorithm == "DCVTD":
            self.model_path = './log/model/' + self.args.env + str(self.args.num_agents) + '/' + self.args.algorithm + "_control_" + str(self.args.control_num_agents) + '/agent_' + str(self.agent_id)
        else:
            self.model_path = './log/model/' + self.args.env + str(self.args.num_agents) + '/' + self.args.algorithm + '/agent_' + str(self.agent_id)
        if args.cuda:
            self.eval_net.cuda()
            self.target_net.cuda()
        if self.args.load:
            self.load_model(1)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.args = args
        self.action_num = action_num
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()

    def init_hidden(self):
        pass

    def choose_action(self, obs, episolon):
        obsnp = np.array(obs)
        obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0),0)
        obs = obs.float()
        if self.args.cuda:
            obs = obs.cuda()
        with torch.no_grad():
            action_value = self.eval_net(obs)
            action_value = action_value.squeeze(dim=0)
            action = torch.max(action_value, 1)[1].cpu().data.numpy()
            if np.random.randn() <= episolon:
                action = action[0]
            else:
                action = np.random.randint(0, self.action_num)
        return action

    def learn(self, episode_data, agent_id, train_round):
        # update the parameters
        if self.learn_step_counter % self.args.target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.args.save and self.learn_step_counter % 100 == 0:
            self.save_model(train_round, int(self.learn_step_counter/100))

        self.learn_step_counter+=1

        batch_state = torch.from_numpy(episode_data['o'][:,:,agent_id,...]).float()
        batch_action = torch.from_numpy(episode_data['u'][:,:,agent_id,...]).long()
        batch_reward = torch.from_numpy(episode_data['r'][:,:,agent_id,...]).float()
        batch_next_state = torch.from_numpy(episode_data['o_next'][:,:,agent_id,...]).float()
        #
        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        #print(batch_state.shape, batch_action.shape, batch_reward.shape, batch_next_state.shape)
        q_eval = self.eval_net(batch_state).gather(2, batch_action)
        q_next = self.target_net(batch_next_state).detach()

        if self.args.double_dqn:
            q_target = batch_reward + self.args.gamma * q_next.gather(2, self.eval_net(batch_next_state).max(2)[1].unsqueeze(dim=2))
            q_target = q_target.detach()
        else:
            q_target = batch_reward + self.args.gamma * q_next.max(2)[0].unsqueeze(dim=2)
            q_target = q_target.detach()
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()
        return loss

    def save_model(self,train_round, num):
        if not os.path.exists(self.model_path + '/' + str(train_round)):
            os.makedirs(self.model_path + '/' + str(train_round))
        torch.save(self.eval_net.state_dict(), self.model_path + '/' + str(train_round) + '/' + str(num) + '_Q_Net.pkl')

    def load_model(self,train_round):
        if os.path.exists(self.model_path + '/' + str(train_round)):
            file_list = sorted(os.listdir(self.model_path + '/' + str(train_round)))
            self.eval_net.load_state_dict(torch.load(self.model_path + '/' + str(train_round) + file_list[-1], map_location='cpu'))
            print('Agent {} successfully loaded {}'.format(self.agent_id, file_list[-1]))
