from torch.distributions.categorical import Categorical
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, N_actions, device):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.N_actions = N_actions
        self.device = device

    
    def add(self, id, state, actions, action_number, reward, next_state):
        data = (id, state,  actions, action_number, reward, next_state)
        self.buffer.append(data)

    
    def __len__(self):
        return len(self.buffer)


    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        id = np.stack([x[0] for x in data])
        state = torch.cat([x[1].unsqueeze(0) for x in data], dim=0)
        actions = torch.cat([x[2].unsqueeze(0) for x in data], dim=0)
        action_number = np.stack([x[3] for x in data])
        reward = np.stack([x[4] for x in data])
        next_state = torch.cat([x[5].unsqueeze(0) for x in data], dim=0)

        return id, state, actions, action_number, reward, next_state


class Actor(nn.Module):
    
    def __init__(self, N_action):
        super(Actor, self).__init__()
        self.N_action = N_action
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv1.bias)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.bias)
        self.pool2 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(9*9, 9)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(9, self.N_action)
        nn.init.zeros_(self.fc2.bias)
    

    def get_action(self, obs):
        out1 = self.pool1(F.selu(self.conv1(obs)))
        out2 = self.pool2(F.selu(self.conv2(out1)))
        out3 = out2.view(-1, 9*9)
        out4 = F.selu(self.fc1(out3))
        out5 = self.fc2(out4)
        out6 = F.softmax(out5, dim=1)
        
        return out6


class Critic(nn.Module):

    def __init__(self, N_action, num_clinet):
        super(Critic, self).__init__()
        self.N_action = N_action
        self.N_client = num_clinet
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv1.bias)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.bias)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv3.bias)
        self.pool3 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(num_clinet*N_action, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)

        self.fc4 = nn.Linear(2*3*3+64, 32)
        self.fc5 = nn.Linear(32, 8)
        self.fc6 = nn.Linear(8, 1)
        

    def get_value(self, S, A):
        out1_s = self.pool1(F.selu(self.conv1(S)))
        out2_s = self.pool2(F.selu(self.conv2(out1_s)))
        out3_s = self.pool3(F.selu(self.conv3(out2_s)))
        out4_s = out3_s.view(-1, 2*3*3)
        
        out1_a = F.selu(self.fc1(A))
        out2_a = F.selu(self.fc2(out1_a))
        out3_a = F.selu(self.fc3(out2_a))

        out4 = torch.cat([out4_s, out3_a], 1)
        out5 = F.selu(self.fc4(out4))
        out6 = F.selu(self.fc5(out5))
        out7 = self.fc6(out6)

        return out7


class V_Net(nn.Module):

    def __init__(self):
        super(V_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv1.bias)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.bias)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv3.bias)
        self.pool3 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(2*3*3, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)


    def get_value(self, S):
        out1 = self.pool1(F.selu(self.conv1(S)))
        out2 = self.pool2(F.selu(self.conv2(out1)))
        out3 = self.pool3(F.selu(self.conv3(out2)))
        out4 = out3.view(-1, 2*3*3)

        out5 = F.selu(self.fc1(out4))
        out6 = F.selu(self.fc2(out5))
        out7 = self.fc3(out6)

        return out7


class COMA:
    
    def __init__(self, N_action, num_agent, buffer_size, batch_size, device):
        self.N_action = N_action
        self.num_agent = num_agent
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.actor = Actor(self.N_action)
        self.critic = Critic(self.N_action, num_agent)
        self.V_net = V_Net()
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, N_actions=self.N_action, device=self.device)


        if self.device == 'cuda':
            self.actor.cuda()
            self.critic.cuda()
            self.V_net.cuda()

        self.gamma = 0.95
        self.critic_loss_fn = torch.nn.MSELoss()
        self.V_net_loff_fn = torch.nn.MSELoss()


    def get_acction(self, obs, env, train_flag, pretrain_flag):        
        obs_tensor = torch.FloatTensor(obs)
        obs_tensor = obs_tensor.to(self.device)

        pi = self.actor.get_action(obs_tensor)
        
        if train_flag:
            clients = env.clients
            actions = np.ones(self.num_agent)*-1
            if pretrain_flag:
                edges = env.all_edge

                for i in range(self.num_agent):
                    client = clients[i]
                    min_idx = 0
                    min_distance = 100000000
                
                    for j in range(len(edges)):
                                edge = edges[j]
                                distance = math.sqrt(pow(client.x - edge.x, 2) + pow(client.y - edge.y, 2))
                                if distance < min_distance:
                                    min_distance = distance
                                    min_idx = j
                    
                    if client.pub_topic[0] == 1:
                        actions[i] = min_idx
            else:
                for i in range(self.num_agent):
                    client = clients[i]
                    if client.pub_topic[0] == 1:
                        actions[i] = Categorical(pi[i]).sample().item()
        else:
            clients = env.clients
            actions = np.ones(self.num_agent)*-1

            for i in range(self.num_agent):
                    client = clients[i]
                    if client.pub_topic[0] == 1:
                        actions[i] = torch.argmax(pi[i])

        return actions, pi

    
    def save_model(self, dir_path, iter):
        torch.save(self.actor.state_dict(), dir_path + 'actor_weight' + '_' + str(iter) + '.pth')
        torch.save(self.critic.state_dict(), dir_path + 'critic_weight' + '_' + str(iter) + '.pth')
        torch.save(self.V_net.state_dict(), dir_path + 'v_net_weight' + '_' + str(iter) + '.pth')


    def load_model(self, dir_path, iter):
        self.actor.load_state_dict(torch.load(dir_path + 'actor_weight' + '_' + str(iter) + '.pth'))
        self.critic.load_state_dict(torch.load(dir_path + 'critic_weight' + '_' + str(iter) + '.pth'))
        self.V_net.load_state_dict(torch.load(dir_path + 'v_net_weight' + '_' + str(iter) + '.pth'))


    def train(self, obs, actions, pi, reward, next_obs):
        # 行動のtensor化
        actions_onehot = torch.zeros(self.num_agent*self.N_action, device=self.device)
        
        for i in range(self.num_agent):
            action = int(actions[i])
            if action != -1:
                actions_onehot[i*self.N_action + action] = 1
        
        # 経験再生用バッファへの追加
        obs_tensor = torch.FloatTensor(obs[0][1:]).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs[0][1:]).to(self.device)

        for i in range(self.num_agent):
            if actions[i] != -1:
                state = obs_tensor
                next_state = next_obs_tensor

                self.replay_buffer.add(i, state, actions_onehot, actions, reward, next_state)

        if len(self.replay_buffer) < self.buffer_size:
            return

        # オプティマイザーの設定
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        V_net_optimizer = torch.optim.Adam(self.V_net.parameters(), lr=1e-3)

        id_exp, obs_exp, actions_exp, action_number_exp, reward_exp, next_obs_exp = self.replay_buffer.get_batch()

        reward_exp = torch.FloatTensor(reward_exp).unsqueeze(1).to(self.device)

        V_target = reward_exp/100 + self.gamma*self.V_net.get_value(next_obs_exp)
        V = self.V_net.get_value(obs_exp)

        V_net_loss = self.V_net_loff_fn(V_target.detach(), V)

        V_net_optimizer.zero_grad()
        V_net_loss.backward(retain_graph=True)
        V_net_optimizer.step()

        # batch_size*1のQ値をtensorとして取得
        Q = self.critic.get_value(obs_exp, actions_exp)

        # critic ネットワークの更新
        critic_loss = self.critic_loss_fn(V_target.detach(), Q)

        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        critic_optimizer.step()

        actor_loss = torch.FloatTensor([0.0])
        actor_loss = actor_loss.to(self.device)

        critic_obs = obs_tensor.unsqueeze(0)
        critic_action = actions_onehot.unsqueeze(0)

        for i in range(1, self.num_agent):
            critic_obs = torch.cat([critic_obs, obs_tensor.unsqueeze(0)], dim=0)
            critic_action = torch.cat([critic_action, actions_onehot.unsqueeze(0)], dim=0)
        
        Q2 = self.critic.get_value(critic_obs, critic_action)

        Q_tmp = torch.zeros(self.N_action, self.num_agent, device=self.device)
                
        for a in range(self.N_action):
            critic_action_copy = critic_action.clone()

            for i in range(self.num_agent):
                for j in range(self.N_action):
                    critic_action_copy[i][i*self.N_action+j] = 0
                    
                critic_action_copy[i][i*self.N_action + a] = 1
            
            Q_tmp[a] = self.critic.get_value(critic_obs, critic_action_copy).squeeze(1)
        
        Q_tmp = torch.permute(Q_tmp, (1, 0))
        
        A = Q2.squeeze(1) - torch.sum(pi*Q_tmp, 1)
    
        cnt = 0
        for i in range(self.num_agent):
            if actions[i] != -1:
                actor_loss = actor_loss + A[i].item() * torch.log(pi[i][int(actions[i])] + 1e-16)
                cnt += 1

        actor_loss = - actor_loss / cnt

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        

class ActorCritic:
    
    def __init__(self, N_action, num_agent, buffer_size, batch_size, device):
        self.N_action = N_action
        self.num_agent = num_agent
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.actor = Actor(self.N_action)
        self.V_net = V_Net()
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, N_actions=self.N_action, device=self.device)


        if self.device == 'cuda':
            self.actor.cuda()
            self.V_net.cuda()

        self.gamma = 0.95
        self.V_net_loff_fn = torch.nn.MSELoss()


    def get_acction(self, obs, env, train_flag, pretrain_flag):        
        obs_tensor = torch.FloatTensor(obs)
        obs_tensor = obs_tensor.to(self.device)

        pi = self.actor.get_action(obs_tensor)

        if train_flag:
            clients = env.clients
            actions = []
            if pretrain_flag:
                edges = env.all_edge

                for i in range(self.num_agent):
                    client = clients[i]
                    min_idx = 0
                    min_distance = 100000000
                    if client.pub_topic[0] == 1:
                        for j in range(len(edges)):
                            edge = edges[j]
                            distance = math.sqrt(pow(client.x - edge.x, 2) + pow(client.y - edge.y, 2))
                            if distance < min_distance:
                                min_distance = distance
                                min_idx = j
                        
                        actions.append(min_idx)
                    else:
                        actions.append(-1)
            else:
                for i in range(self.num_agent):
                    client = clients[i]
                    if client.pub_topic[0] == 1:
                        actions.append(Categorical(pi[i]).sample().item())
                    else:
                        actions.append(-1)
        else:
            clients = env.clients
            actions = np.ones(self.num_agent)*-1

            for i in range(self.num_agent):
                    client = clients[i]
                    if client.pub_topic[0] == 1:
                        actions[i] = torch.argmax(pi[i])
        
        return actions, pi

    
    def save_model(self, dir_path, iter):
        torch.save(self.actor.state_dict(), dir_path + 'actor_weight' + '_' + str(iter) + '.pth')
        torch.save(self.V_net.state_dict(), dir_path + 'v_net_weight' + '_' + str(iter) + '.pth')


    def load_model(self, dir_path, iter):
        self.actor.load_state_dict(torch.load(dir_path + 'actor_weight' + '_' + str(iter) + '.pth'))
        self.V_net.load_state_dict(torch.load(dir_path + 'v_net_weight' + '_' + str(iter) + '.pth'))


    def train(self, obs, actions, pi, reward, next_obs):
       
        # 行動のtensor化
        actions_onehot = torch.zeros(self.num_agent*self.N_action, device=self.device)
         
        for i in range(self.num_agent):
            action = actions[i]
            if action != -1:
                actions_onehot[i*self.N_action + action] = 1
        
        # 経験再生用バッファへの追加
        obs_tensor = torch.FloatTensor(obs[0][1:]).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs[0][1:]).to(self.device)

        for i in range(self.num_agent):
            if actions[i] != -1:
                state = obs_tensor
                next_state = next_obs_tensor

                self.replay_buffer.add(i, state, actions_onehot, actions, reward, next_state)

        if len(self.replay_buffer) < self.buffer_size:
            return

        # オプティマイザーの設定
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        V_net_optimizer = torch.optim.Adam(self.V_net.parameters(), lr=1e-3)

        id_exp, obs_exp, actions_exp, action_number_exp, reward_exp, next_obs_exp = self.replay_buffer.get_batch()

        reward_exp = torch.FloatTensor(reward_exp).unsqueeze(1).to(self.device)

        V_target = reward_exp/100 + self.gamma*self.V_net.get_value(next_obs_exp)
        V = self.V_net.get_value(obs_exp)

        V_net_loss = self.V_net_loff_fn(V_target.detach(), V)

        V_net_optimizer.zero_grad()
        V_net_loss.backward(retain_graph=True)
        V_net_optimizer.step()

        actor_loss = torch.FloatTensor([0.0])
        actor_loss = actor_loss.to(self.device)

        v_obs = obs_tensor.unsqueeze(0)
        v_next_obs = next_obs_tensor.unsqueeze(0)

        V_target = reward_exp/100 + self.gamma*self.V_net.get_value(v_next_obs)
        V = self.V_net.get_value(v_obs)

        A = V_target - V

        cnt = 0
        for i in range(self.num_agent):
            if actions[i] != -1:
                actor_loss = actor_loss + A[i].item() * torch.log(pi[i][actions[i]] + 1e-16)
                cnt += 1

        actor_loss = - actor_loss / cnt

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        