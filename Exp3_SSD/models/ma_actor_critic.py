import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, args, action_num):
        super(CNN, self).__init__()
        self.args = args
        self.action_num = action_num
        self.cnn = nn.Conv2d(3, 6, kernel_size=3, stride=1)

    def forward(self, x):
        x = F.relu(self.cnn(x))
        return x


class Actor(nn.Module):
    def __init__(self, args, action_num):
        super(Actor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(169 * 6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.action_out = nn.Linear(32, action_num)
        self.action_out.weight.data.normal_(0, 0.1)

    def forward(self, obs, cnn, h0=None, c0=None):
        Batch, seq_len = obs.shape[0], obs.shape[1]
        x_emb = []
        for seq in range(seq_len):
            embedding = obs[:,seq,...]
            embedding = embedding.permute(0,3,1,2)
            embedding = cnn(embedding)
            embedding = embedding.reshape(Batch, -1)
            embedding = F.relu(self.fc1(embedding))
            embedding = F.relu(self.fc2(embedding))
            x_emb.append(embedding)
        x = torch.stack(x_emb.copy(), dim=1)
        if self.args.cuda:
            x = x.cuda()
        if h0 == None:
            x, (h, c) = self.lstm(x)
            x = self.action_out(x)
            return x
        else:
            x, (h, c) = self.lstm(x, (h0, c0))
            x = self.action_out(x)
            return x, h, c

class Critic(nn.Module):
    def __init__(self, args, action_num):
        super(Critic, self).__init__()
        self.agent_num = args.num_agents
        self.action_num = action_num
        self.args = args
        self.fc1 = nn.Linear(169 * 6, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 0.1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.fc3 = nn.Linear(32*self.agent_num + self.action_num*self.agent_num, 32)
        self.fc3.weight.data.normal_(0, 0.1)
        self.q_out = nn.Linear(32, 1)

    def forward(self, s, a, cnn):
        s_embedding = torch.FloatTensor(s.shape[0], s.shape[1], s.shape[2], 32)
        if self.args.cuda:
            s_embedding = s_embedding.cuda()
        for i in range(self.args.num_agents):
            obs = s[:,:,i,...]
            Batch, seq_len = obs.shape[0], obs.shape[1]
            x_emb = []
            for seq in range(seq_len):
                embedding = obs[:, seq, ...]
                embedding = embedding.permute(0, 3, 1, 2)
                embedding = cnn(embedding)
                embedding = embedding.reshape(Batch, -1)
                embedding = F.relu(self.fc1(embedding))
                embedding = F.relu(self.fc2(embedding))
                x_emb.append(embedding)
            x = torch.stack(x_emb.copy(), dim=1)
            if self.args.cuda:
                x = x.cuda()
            x, (h, c) = self.lstm(x)
            s_embedding[:, :, i, :] = x
        s_embedding = s_embedding.reshape(s.shape[0], s.shape[1], -1)
        a = a.reshape(a.shape[0], a.shape[1], -1)
        sa_mix = torch.cat((s_embedding, a), dim=2)
        x = F.relu(self.fc3(sa_mix))
        q_value = self.q_out(x)
        return q_value
