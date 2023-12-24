import torch
import os
from models.ma_actor_critic import Actor, Critic, CNN
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

class MADDPG:
    def __init__(self, args, action_num, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.agent_id = agent_id
        self.args = args
        self.action_num = action_num
        self.train_step = 0
        # create the network
        self.actor_network = Actor(self.args, self.action_num)
        self.critic_network = Critic(self.args, self.action_num)

        # build up the target network
        self.actor_target_network = Actor(self.args, self.action_num)
        self.critic_target_network = Critic(self.args, self.action_num)

        self.obs_embedding_a = CNN(self.args, self.action_num)
        self.obs_embedding_c = CNN(self.args, self.action_num)

        if self.args.cuda:
            self.actor_network.cuda()
            self.actor_target_network.cuda()
            self.critic_network.cuda()
            self.critic_target_network.cuda()
            self.obs_embedding_a.cuda()
            self.obs_embedding_c.cuda()
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_parameters = list(self.actor_network.parameters()) + list(self.obs_embedding_a.parameters())
        self.critic_parameters = list(self.critic_network.parameters()) + list(self.obs_embedding_c.parameters())

        self.actor_optim = torch.optim.Adam(self.actor_parameters, lr=self.args.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_parameters, lr=self.args.critic_lr)

        self.loss_func = torch.nn.MSELoss()

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def init_hidden(self):
        self.h0, self.c0 = torch.randn(1, 1, 32),torch.randn(1, 1, 32)

    def choose_action(self, obs, episolon):
        obsnp = np.array(obs)
        obs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(obsnp.copy()), 0),0)
        obs = obs.float()
        if self.args.cuda:
            obs = obs.cuda()
        with torch.no_grad():
            if self.args.cuda:
                self.h0 = self.h0.cuda()
                self.c0 = self.c0.cuda()
            action_out,self.h0,self.c0 = self.actor_network(obs, self.obs_embedding_a, self.h0, self.c0)
            action_prob = F.gumbel_softmax(action_out, dim=2, hard=True)
            if np.random.randn() <= episolon:
                action = action_prob.max(dim=2)[1].cpu().data.numpy()
                action = action[0][0]
            else:
                action = np.random.randint(0, self.action_num)
        return action

    # update the network
    def learn(self, episode_data, agent_id, agent_maddpg, train_round):
        batch_state = torch.from_numpy(episode_data['o']).float()
        batch_action = torch.from_numpy(episode_data['u']).long()
        batch_action = torch.zeros(self.args.batch_size, self.args.num_steps, self.args.num_agents,
                             self.args.action_num).scatter_(3, batch_action, 1)
        batch_reward = torch.from_numpy(episode_data['r']).float()
        batch_next_state = torch.from_numpy(episode_data['o_next']).float()

        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        state = batch_state
        action = batch_action
        reward = batch_reward
        next_state = batch_next_state

        a_next = []
        with torch.no_grad():
            for i in range(self.args.num_agents):
                if i == agent_id:
                    a_next.append(self.actor_target_network(next_state[:,:,i,...], self.obs_embedding_a))
                else:
                    a_next.append(agent_maddpg[i].actor_target_network(next_state[:,:,i,...], agent_maddpg[i].obs_embedding_a))
            a_next = torch.stack(a_next, dim=2)
            if self.args.cuda:
                a_next = a_next.cuda()
        q_next = self.critic_target_network(next_state, a_next, self.obs_embedding_c).detach()
        q_value = self.critic_network(state, action, self.obs_embedding_c)
        q_target = (reward[:,:,agent_id,...] + self.args.gamma * q_next).detach()

        critic_loss = self.loss_func(q_value, q_target)

        for i in range(self.args.num_agents):
            a_prob = agent_maddpg[i].actor_network(state[:, :, i, ...], agent_maddpg[i].obs_embedding_a)
            action[:, :, i, ...] = F.gumbel_softmax(a_prob, dim=2)
        actor_loss = -self.critic_network(state, action, self.obs_embedding_c).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.args.grad_norm_clip)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optim.step()

        if self.train_step % self.args.replace_param == 0:
            self._soft_update_target_network()
        self.train_step += 1

        return critic_loss, actor_loss
