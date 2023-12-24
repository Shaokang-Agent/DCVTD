import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Obs_embedding(nn.Module):
    """sharing for obs embedding"""
    def __init__(self):
        super(Obs_embedding, self).__init__()
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.cnn.weight.data.normal_(0, 0.1)

    def forward(self, obs):
        obs = F.leaky_relu(self.cnn(obs))
        return obs

class QNet(nn.Module):
    """docstring for Net"""
    def __init__(self, action_num, args):
        super(QNet, self).__init__()
        self.args = args
        self.action_num = action_num
        self.fc1 = nn.Linear(169*6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32, action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, obs, obs_feature):
        Batch, seq_len = obs.shape[0], obs.shape[1]
        x = obs.permute(0, 1, 4, 2, 3)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = obs_feature(x)
        x = x.reshape(Batch, seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x

class CQNet(nn.Module):
    """docstring for Net"""
    def __init__(self, action_num, args):
        super(CQNet, self).__init__()
        self.action_num = action_num
        self.args = args
        self.fc1 = nn.Linear(169*6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(32*2+self.action_num, self.action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, obs, other_obs, other_act, obs_feature):
        Batch, seq_len = obs.shape[0], obs.shape[1]
        x = obs.permute(0, 1, 4, 2, 3)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = obs_feature(x)
        x = x.reshape(Batch, seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        y = other_obs.permute(0, 1, 4, 2, 3)
        y = torch.flatten(y, start_dim=0, end_dim=1)
        y = obs_feature(y)
        y = y.reshape(Batch, seq_len, -1)
        y = F.leaky_relu(self.fc1(y))
        y = F.leaky_relu(self.fc2(y))

        action = F.one_hot(other_act, self.action_num).squeeze(dim=2)
        z = torch.cat([x, y, action], dim=2)
        z = self.out(z)

        return z

class Detection(nn.Module):
    """docstring for Net"""
    def __init__(self, args, action_num):
        super(Detection, self).__init__()
        self.args = args
        self.action_num = action_num
        self.fc1 = nn.Linear(169*6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)

        self.key = nn.Linear(self.args.hidden_dim, self.args.dim)
        self.key.weight.data.normal_(0, 0.1)
        self.select = nn.Linear(self.args.hidden_dim, self.args.dim)
        self.select.weight.data.normal_(0, 0.1)
        self.keys = nn.ModuleList()
        self.selects = nn.ModuleList()
        for i in range(self.args.heads):
            self.keys.append(self.key)
            self.selects.append(self.select)

    def forward(self, self_state, other_state, difference, obs_feature):
        Batch, seq_len = self_state.shape[0], self_state.shape[1]
        s_embeddings = torch.FloatTensor(Batch,seq_len, self.args.num_agents-1, self.args.hidden_dim)
        for i in range(self.args.num_agents-1):
            other_obs = other_state[:,:,i,...]
            x = other_obs.permute(0, 1, 4, 2, 3)
            x = torch.flatten(x, start_dim=0, end_dim=1)
            x = obs_feature(x)
            x = x.reshape(Batch, seq_len, -1)
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            s_embeddings[:,:,i,:] = x

        self_embedding = self_state.permute(0, 1, 4, 2, 3)
        self_embedding = torch.flatten(self_embedding, start_dim=0, end_dim=1)
        self_embedding = obs_feature(self_embedding)
        self_embedding = self_embedding.reshape(Batch, seq_len, -1)
        self_embedding = F.leaky_relu(self.fc1(self_embedding))
        self_embedding = F.leaky_relu(self.fc2(self_embedding))

        keys = torch.FloatTensor(self.args.heads, self.args.batch_size, self.args.num_steps, self.args.num_agents-1, self.args.dim)
        selectors = torch.FloatTensor(self.args.heads, self.args.batch_size, self.args.num_steps, self.args.num_agents-1, self.args.dim)
        Threat_coefficent_Head = torch.FloatTensor(self.args.heads, self.args.batch_size, self.args.num_steps, self.args.num_agents-1)
        if self.args.cuda:
            keys = keys.cuda()
            selectors = selectors.cuda()
            Threat_coefficent_Head = Threat_coefficent_Head.cuda()
            s_embeddings = s_embeddings.cuda()
        for H in range(self.args.heads):
            for i in range(self.args.num_agents-1):
                keys[H,:,:,i,:] = self.keys[H](self_embedding)
                selectors[H,:,:,i,:] = self.selects[H](s_embeddings[:,:,i,:])
                Threat_coefficent_Head[H,:,:,i] = torch.matmul(torch.unsqueeze(keys[H,:,:,i,:], dim=2), torch.unsqueeze(selectors[H,:,:,i,:], dim=3)).squeeze() / (16*np.sqrt((self.args.num_agents-1)*self.args.batch_size*self.args.num_steps*self.args.dim))
        Threat = torch.mean(F.gumbel_softmax(Threat_coefficent_Head, dim=3), dim=0)
        internal_reward = torch.sum(torch.mul(Threat, difference), dim=2).unsqueeze(dim=2)
        return internal_reward, Threat

class DCQF():
    """docstring for DQN"""
    def __init__(self, args, action_num, agent_id):
        super(DCQF, self).__init__()
        self.agent_id = agent_id
        self.args = args
        self.obs_feature = Obs_embedding()
        self.eval_net, self.target_net = QNet(action_num, args).float(), QNet(action_num, args).float()
        self.ceval_net, self.ctarget_net = CQNet(action_num, args).float(), CQNet(action_num, args).float()

        self.model_path = './log/model/' + self.args.env + str(
            self.args.num_agents) + '/' + self.args.algorithm + "_control_" + str(self.args.control_num_agents) + '/agent_' + str(self.agent_id)
        if self.args.load:
            self.load_model(1)
        self.ctarget_net.load_state_dict(self.ceval_net.state_dict())
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.action_num = action_num
        self.internal_net = Detection(self.args, action_num).float()

        if self.args.cuda:
            self.eval_net.cuda()
            self.target_net.cuda()
            self.ceval_net.cuda()
            self.ctarget_net.cuda()
            self.obs_feature.cuda()
            self.internal_net.cuda()
        self.learn_step_counter = 0
        self.eval_parameters = list(self.eval_net.parameters()) + list(self.ceval_net.parameters()) + list(self.internal_net.parameters()) + list(self.obs_feature.parameters())
        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.args.lr)
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
            action_value = self.eval_net(obs,self.obs_feature)
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
            self.ctarget_net.load_state_dict(self.ceval_net.state_dict())

        if self.args.save and self.learn_step_counter % 100 == 0:
            self.save_model(train_round, int(self.learn_step_counter/100))

        self.learn_step_counter+=1

        ENV_ALPHA = np.min([self.args.env_alpha_final, self.args.env_alpha_initial + (self.learn_step_counter / self.args.epsilon_episode) * (self.args.env_alpha_final - self.args.env_alpha_initial)])

        batch_state = torch.from_numpy(episode_data['o']).float()
        batch_action = torch.from_numpy(episode_data['u']).long()
        batch_reward = torch.from_numpy(episode_data['r']).float()
        batch_next_state = torch.from_numpy(episode_data['o_next']).float()

        if self.args.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()


        state = batch_state[:,:,agent_id,...]
        otherstate = torch.cat( (batch_state[:,:,:agent_id,...], batch_state[:,:,agent_id+1:,...]), dim=2 )
        action = batch_action[:,:,agent_id,...]
        otheraction = torch.cat( (batch_action[:,:,:agent_id,...], batch_action[:,:,agent_id+1:,...]), dim=2 )
        reward = batch_reward[:,:,agent_id,...]
        next_state = batch_next_state[:,:,agent_id,...]
        other_next_state = torch.cat( (batch_next_state[:,:,:agent_id,...], batch_next_state[:,:,agent_id+1:,...]), dim=2 )

        #print(batch_state.shape, batch_otherstate.shape, batch_otheraction.shape)
        q_eval = self.eval_net(state, self.obs_feature).gather(1, action)
        q_next = self.target_net(next_state, self.obs_feature).detach()


        cq_eval = torch.FloatTensor(self.args.batch_size, self.args.num_steps, self.args.num_agents-1, 1)
        cq_next = torch.FloatTensor(self.args.batch_size, self.args.num_steps, self.args.num_agents-1, self.action_num)
        cq_target = torch.FloatTensor(self.args.batch_size, self.args.num_steps, self.args.num_agents-1, 1)
        q_cq_difference = torch.FloatTensor(self.args.batch_size, self.args.num_steps, self.args.num_agents-1)

        if self.args.cuda:
            cq_eval = cq_eval.cuda()
            cq_next = cq_next.cuda()
            cq_target = cq_target.cuda()
            q_cq_difference = q_cq_difference.cuda()

        for num in range(self.args.num_agents-1):
            cq_eval[:,:, num,:] = self.ceval_net(state, otherstate[:,:,num,...], otheraction[:,:,num,:],self.obs_feature).gather(2, action)
            q_other = self.eval_net(otherstate[:,:,num,...], self.obs_feature).max(2)[0].detach()
            q_other_probability = F.gumbel_softmax(self.eval_net(otherstate[:,:,num,...], self.obs_feature), dim=2, hard=False)
            q_other_advatage = q_other - torch.mean(q_other_probability * self.eval_net(otherstate[:,:,num,...], self.obs_feature), dim=2)
            q_other_advatage = q_other_advatage.detach()

            next_other_action = self.eval_net(other_next_state[:, :, num, ...], self.obs_feature).max(2)[1].view(self.args.batch_size, self.args.num_steps,1).detach()
            cq_next[:,:,num,:] = self.ctarget_net(next_state, other_next_state[:, :, num, ...], next_other_action, self.obs_feature).detach()
            cq_other = self.ceval_net(otherstate[:,:,num,...], state, action, self.obs_feature).max(2)[0].detach()
            cq_other_probability = F.gumbel_softmax(self.ceval_net(otherstate[:,:,num,...], state, action, self.obs_feature), dim=2, hard=False)
            cq_other_advatage = cq_other - torch.mean(cq_other_probability * self.ceval_net(otherstate[:,:,num,...], state, action, self.obs_feature), dim=2)
            cq_other_advatage = cq_other_advatage.detach()

            if self.args.double_dqn:
                cq_target[:,:,num,:] = reward + self.args.gamma * cq_next[:,:,num,:].gather(2, self.eval_net(
                    next_state, self.obs_feature).max(2)[1].view(self.args.batch_size, self.args.num_steps,1)).reshape(self.args.batch_size, self.args.num_steps,1)
            else:
                cq_target[:,:,num,:] = reward + self.args.gamma * cq_next[:,:,num,:].max(2)[0].reshape(self.args.batch_size, self.args.num_steps,1)

            q_cq_difference[:,:,num] = q_other_advatage - cq_other_advatage

        if self.learn_step_counter == 1:
            internal_reward = torch.FloatTensor(self.args.batch_size, self.args.num_steps, 1).zero_()
            if self.args.cuda:
                internal_reward = internal_reward.cuda()
        else:
            internal_reward, threat = self.internal_net(state, otherstate, q_cq_difference, self.obs_feature)
            print("agent {}, q_cq_difference: {}".format(agent_id, torch.mean(q_cq_difference[:5], dim=1)))
            internal_reward = internal_reward * self.args.internal_scale
            print("agent {}, internal_reward: {}".format(agent_id, torch.mean(internal_reward[:5], dim=1)))
            internal_reward = torch.clamp(internal_reward, -0.5, 0.5)


        if self.args.double_dqn:
            q_target = 1 * reward + (1-ENV_ALPHA)*internal_reward + self.args.gamma * q_next.gather(1, self.eval_net(next_state, self.obs_feature).max(2)[1].view(self.args.batch_size, self.args.num_steps,1)).view(self.args.batch_size, self.args.num_steps,1)
        else:
            q_target = 1 * reward + (1-ENV_ALPHA)*internal_reward + self.args.gamma * q_next.max(2)[0].view(self.args.batch_size, self.args.num_steps,1)

        cq_target = cq_target.detach()

        loss1 = self.loss_func(q_eval, q_target)
        loss2 = self.loss_func(cq_eval, cq_target)
        loss = loss1 + loss2
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model(self,train_round,num):
        if not os.path.exists(self.model_path + '/' + str(train_round)):
            os.makedirs(self.model_path + '/' + str(train_round))
        torch.save(self.eval_net.state_dict(), self.model_path + '/' + str(train_round) + '/' + str(num) + '_Q_Net.pkl')
        torch.save(self.ceval_net.state_dict(), self.model_path + '/' + str(train_round) + '/' + str(num) + '_CQ_Net.pkl')
        torch.save(self.internal_net.state_dict(), self.model_path + '/' + str(train_round) + '/' + str(num) + '_Internal_Net.pkl')
        torch.save(self.obs_feature.state_dict(), self.model_path + '/' + str(train_round) + '/' + str(num) + '_Obs_Feature_Net.pkl')

    def load_model(self,train_round):
        if os.path.exists(self.model_path + '/' + str(train_round)):
            self.eval_net.load_state_dict(torch.load(self.model_path + '/' + str(train_round)  + '/' + str(49) + '_Q_Net.pkl', map_location='cpu'))
            self.ceval_net.load_state_dict(torch.load(self.model_path + '/' + str(train_round)  + '/' + str(49) + '_CQ_Net.pkl', map_location='cpu'))
            self.internal_net.load_state_dict(torch.load(self.model_path + '/' + str(train_round)  + '/' + str(49) + '_Internal_Net.pkl', map_location='cpu'))
            self.obs_feature.load_state_dict(torch.load(self.model_path + '/' + str(train_round)  + '/' + str(49) + '_Obs_Feature_Net.pkl', map_location='cpu'))
            print('Agent {} successfully loaded {}'.format(self.agent_id, self.model_path + '/' + str(train_round)))
