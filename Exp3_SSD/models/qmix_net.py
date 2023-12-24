import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    """sharing for obs embedding"""
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.cnn.weight.data.normal_(0, 0.1)

    def forward(self, obs):
        obs = F.relu(self.cnn(obs))
        return obs

class QNet(nn.Module):
    """docstring for Net"""
    def __init__(self, args, action_num):
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
        x = F.leaky_relu(self.cnn(x))
        x = x.reshape(Batch, seq_len, -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(169 * 6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)

        if args.two_hyper_layers:
            self.hyper_w1_1 = nn.Linear(self.args.num_agents*32, args.hyper_hidden_dim)
            self.hyper_w1_1.weight.data.normal_(0, 0.1)
            self.hyper_w1_2 = nn.Linear(args.hyper_hidden_dim, self.args.num_agents*args.qmix_hidden_dim)
            self.hyper_w1_2.weight.data.normal_(0, 0.1)
            self.hyper_w1 = nn.Sequential(self.hyper_w1_1,
                                          nn.ReLU(),
                                          self.hyper_w1_2)

            self.hyper_w2_1 = nn.Linear(self.args.num_agents*32, args.hyper_hidden_dim)
            self.hyper_w2_1.weight.data.normal_(0, 0.1)
            self.hyper_w2_2 = nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim)
            self.hyper_w2_2.weight.data.normal_(0, 0.1)
            self.hyper_w2 = nn.Sequential(self.hyper_w2_1,
                                          nn.ReLU(),
                                          self.hyper_w2_2)
        else:
            self.hyper_w1 = nn.Linear(self.args.num_agents*32, self.args.num_agents*args.qmix_hidden_dim)
            self.hyper_w1.weight.data.normal_(0, 0.1)
            self.hyper_w2 = nn.Linear(self.args.num_agents*32, args.qmix_hidden_dim)
            self.hyper_w2.weight.data.normal_(0, 0.1)

        self.hyper_b1 = nn.Linear(self.args.num_agents*32, args.qmix_hidden_dim)
        self.hyper_b1.weight.data.normal_(0, 0.1)

        self.hyper_b2_1 = nn.Linear(self.args.num_agents*32, args.qmix_hidden_dim)
        self.hyper_b2_1.weight.data.normal_(0, 0.1)
        self.hyper_b2_2 = nn.Linear(args.qmix_hidden_dim, 1)
        self.hyper_b2_2.weight.data.normal_(0, 0.1)
        self.hyper_b2 = nn.Sequential(self.hyper_b2_1,
                                     nn.ReLU(),
                                     self.hyper_b2_2)

    def forward(self, q_values, s, obs_feature):
        Batch, seq_len = s.shape[0], s.shape[1]
        q_values = q_values.reshape(q_values.shape[0], q_values.shape[1], -1, self.args.num_agents)
        s_embedding = torch.zeros([s.shape[0], s.shape[1], s.shape[2], 32])
        if self.args.cuda:
            s_embedding = s_embedding.cuda()
        for i in range(self.args.num_agents - 1):
            obs = s[:, :, i, ...]
            x = obs.permute(0, 1, 4, 2, 3)
            x = torch.flatten(x, start_dim=0, end_dim=1)
            x = F.leaky_relu(self.cnn(x))
            x = x.reshape(Batch, seq_len, -1)
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            s_embedding[:, :, i, :] = x
        s_embedding = s_embedding.reshape(s.shape[0], s.shape[1], -1)

        w1 = torch.abs(self.hyper_w1(s_embedding))
        b1 = self.hyper_b1(s_embedding)
        w1 = w1.view(Batch, seq_len, self.args.num_agents, self.args.qmix_hidden_dim)
        b1 = b1.view(Batch, seq_len, 1, self.args.qmix_hidden_dim)

        hidden = F.elu(torch.matmul(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(s_embedding)).unsqueeze(dim=3)
        b2 = self.hyper_b2(s_embedding)

        q_total = torch.matmul(hidden, w2).squeeze(dim=2) + b2
        q_total = q_total.reshape(Batch, seq_len, 1)
        return q_total