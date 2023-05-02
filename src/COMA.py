from torch.distributions.categorical import Categorical
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import time as time_modu


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, N_actions, device):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.N_actions = N_actions
        self.device = device

    
    def add(self, state, state_topic, actions_onehot, reward, next_state, next_state_topic):
        data = (state, state_topic, actions_onehot, reward, next_state, next_state_topic)
        self.buffer.append(data)

    
    def __len__(self):
        return len(self.buffer)


    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.cat([x[0].unsqueeze(0) for x in data], dim=0)
        state_topic = torch.cat([x[1].unsqueeze(0) for x in data], dim=0)
        actions_onehot = torch.cat([x[2].unsqueeze(0) for x in data], dim=0)
        reward = np.stack([x[3] for x in data])
        next_state = torch.cat([x[4].unsqueeze(0) for x in data], dim=0)
        next_state_topic = torch.cat([x[5].unsqueeze(0) for x in data], dim=0)

        return state, state_topic, actions_onehot, reward, next_state, next_state_topic


class Actor(nn.Module):
    
    def __init__(self, N_action):
        super(Actor, self).__init__()
        self.N_action = N_action
        self.batch_norm2d_1 = nn.BatchNorm2d(9)
        self.batch_norm2d_2 = nn.BatchNorm2d(9)
        self.batch_norm2d_3 = nn.BatchNorm2d(4)
        self.batch_norm2d_4 = nn.BatchNorm2d(4)
        self.batch_norm1d_1 = nn.BatchNorm1d(3)
        self.batch_norm1d_2 = nn.BatchNorm1d(126)
        self.batch_norm1d_3 = nn.BatchNorm1d(64)

        self.conv1 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)

        self.pool1 = nn.MaxPool2d(3)
        self.pool2 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(4*9*9 + 3, 126)
        self.fc2 = nn.Linear(126, 64)
        self.fc3 = nn.Linear(64, self.N_action)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    

    def get_action(self, obs, obs_topic):
        out1 = self.batch_norm2d_1(obs)
        out2 = F.selu(self.batch_norm2d_2(self.conv1(out1)))
        out3 = F.selu(self.batch_norm2d_3(self.conv2(out2)))
        out4 = self.pool1(out3)
        out5 = F.selu(self.batch_norm2d_4(self.conv3(out4)))
        out6 = self.pool2(out5)
        out7 = out6.view(-1, 4*9*9)

        out8 = self.batch_norm1d_1(obs_topic)
        out9 = torch.cat([out7, out8], 1)
        out10 = F.selu(self.batch_norm1d_2(self.fc1(out9)))
        out11 = F.selu(self.batch_norm1d_3(self.fc2(out10)))
        out12 = self.fc3(out11)
        out12 = F.softmax(out12, dim=1)
        
        return out12


class Critic(nn.Module):

    def __init__(self, N_action, num_client, num_topic):
        super(Critic, self).__init__()
        self.N_action = N_action
        self.N_client = num_client
        self.N_topic = num_topic

        self.batch_norm2d_1 = nn.BatchNorm2d(self.N_topic*4 + 2)
        self.batch_norm2d_2 = nn.BatchNorm2d(self.N_topic*4 + 2)
        self.batch_norm2d_3 = nn.BatchNorm2d(4)
        self.batch_norm2d_4 = nn.BatchNorm2d(2)
        self.batch_norm_topic = nn.BatchNorm1d(3*self.N_topic)
        self.batch_norm_action1 = nn.BatchNorm1d(512)
        self.batch_norm_action2 = nn.BatchNorm1d(256)
        self.batch_norm_action3 = nn.BatchNorm1d(64)
        self.batch_norm1d_1 = nn.BatchNorm1d(128)
        self.batch_norm1d_2 = nn.BatchNorm1d(64)


        self.conv1 = nn.Conv2d(in_channels=self.N_topic*4+2, out_channels=self.N_topic*4+2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.N_topic*4+2, out_channels=4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)

        self.pool1 = nn.MaxPool2d(3)
        self.pool2 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(self.N_topic*self.N_client*self.N_action, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)

        self.fc4 = nn.Linear(2*9*9+3*self.N_topic+64, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        

    def get_value(self, S, S_topic, A):
        out1_s = self.conv1(self.batch_norm2d_1(S))
        out2_s = self.pool1(F.selu(self.batch_norm2d_2(out1_s)))
        out3_s = self.pool2(F.selu(self.batch_norm2d_3(self.conv2(out2_s))))
        out4_s = F.selu(self.batch_norm2d_4(self.conv3(out3_s)))
        out5_s = out4_s.view(-1, 2*9*9)

        out_topic = self.batch_norm_topic(S_topic)
        
        out1_a = F.selu(self.batch_norm_action1(self.fc1(A)))
        out2_a = F.selu(self.batch_norm_action2(self.fc2(out1_a)))
        out3_a = F.selu(self.batch_norm_action3(self.fc3(out2_a)))
        
        out4 = torch.cat([out5_s, out_topic, out3_a], 1)
        out5 = F.selu(self.batch_norm1d_1(self.fc4(out4)))
        out6 = F.selu(self.batch_norm1d_2(self.fc5(out5)))
        out7 = self.fc6(out6)

        return out7


class V_Net(nn.Module):

    def __init__(self, num_topic):
        super(V_Net, self).__init__()
        self.N_topic = num_topic

        self.batch_norm2d_1 = nn.BatchNorm2d(self.N_topic*4 + 2)
        self.batch_norm2d_2 = nn.BatchNorm2d(self.N_topic*4 + 2)
        self.batch_norm2d_3 = nn.BatchNorm2d(4)
        self.batch_norm2d_4 = nn.BatchNorm2d(2)
        self.batch_norm_topic = nn.BatchNorm1d(3*self.N_topic)
        self.batch_norm1d_1 = nn.BatchNorm1d(64)
        self.batch_norm1d_2 = nn.BatchNorm1d(16)

        self.conv1 = nn.Conv2d(in_channels=self.N_topic*4+2, out_channels=self.N_topic*4+2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.N_topic*4+2, out_channels=4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)

        self.pool1 = nn.MaxPool2d(3)
        self.pool2 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(2*9*9 + 3*self.N_topic, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)


    def get_value(self, S, S_topic):
        out1 = self.conv1(self.batch_norm2d_1(S))
        out2 = self.pool1(F.selu(self.batch_norm2d_2(out1)))
        out3 = self.pool2(F.selu(self.batch_norm2d_3(self.conv2(out2))))
        out4 = F.selu(self.batch_norm2d_4(self.conv3(out3)))

        out5 = out4.view(-1, 2*9*9)

        out_topic = self.batch_norm_topic(S_topic)

        out6 = torch.cat([out5, out_topic], dim=1)

        out7 = F.selu(self.batch_norm1d_1(self.fc1(out6)))
        out8 = F.selu(self.batch_norm1d_2(self.fc2(out7)))
        out9 = self.fc3(out8)

        return out9


class COMA:
    
    def __init__(self, N_action, num_agent, num_topic, buffer_size, batch_size, eps_clip, device):
        self.N_action = N_action
        self.num_agent = num_agent
        self.num_topic = num_topic
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.device = device
        self.actor = Actor(self.N_action)
        self.old_actor = Actor(self.N_action)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.N_action, num_agent, num_topic)
        self.target_critic = Critic(self.N_action, num_agent, num_topic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.V_net = V_Net(num_topic)
        self.target_V_net = V_Net(num_topic)
        self.target_V_net.load_state_dict(self.V_net.state_dict())
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, N_actions=self.N_action, device=self.device)

        # オプティマイザーの設定
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.V_net_optimizer = torch.optim.Adam(self.V_net.parameters(), lr=1e-3)

        if self.device != 'cpu':
            self.actor.cuda(self.device)
            self.critic.cuda(self.device)
            self.V_net.cuda(self.device)
            self.old_actor.cuda(self.device)
            self.target_critic.cuda(self.device)
            self.target_V_net.cuda(self.device)

        self.gamma = 0.95
        self.critic_loss_fn = torch.nn.MSELoss()
        self.V_net_loff_fn = torch.nn.MSELoss()


    def get_acction(self, obs, obs_topic, env, train_flag, pretrain_flag):        
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        obs_topic_tensor = torch.FloatTensor(obs_topic).to(self.device)

        obs_tensor = obs_tensor.permute(1, 0, 2, 3, 4)

        pi = torch.zeros((self.num_topic, self.num_agent, self.N_action)).to(self.device)
        pi_old = torch.zeros((self.num_topic, self.num_agent, self.N_action)).to(self.device)

        for t in range(self.num_topic):
            obs_topic_tensor_tmp = obs_topic_tensor[t].unsqueeze(0)
            for _ in range(self.num_agent-1):
                obs_topic_tensor_tmp = torch.cat([obs_topic_tensor_tmp, obs_topic_tensor[t].unsqueeze(0)], 0)
            
            pi[t] = self.actor.get_action(obs_tensor[t], obs_topic_tensor_tmp)
            pi_old[t] = self.old_actor.get_action(obs_tensor[t], obs_topic_tensor_tmp)

        actions = torch.full((self.num_topic, self.num_agent), -1, device=self.device)
        
        clients = env.clients

        if train_flag:
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
                    
                    for t in range(self.num_topic):
                        if client.pub_topic[t] == 1:
                            actions[t][i] = min_idx
            else:
                pub_topic_tensor = torch.stack([torch.tensor(client.pub_topic) for client in clients]).T
                mask = pub_topic_tensor.bool()  # マスクを作成
                actions[mask] = Categorical(pi[mask]).sample()  # マスクを使ってアクションをサンプリング
        else:
            for i in range(self.num_agent):
                client = clients[i]
                for t in range(self.num_topic):
                    if client.pub_topic[t] == 1:
                        actions[t][i] = torch.argmax(pi[t][i])

        return actions, pi, pi_old
    

    def old_net_update(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    
    def save_model(self, dir_path, actor_weight, critic_weight, v_net_weight, iter):
        torch.save(self.actor.state_dict(), dir_path + actor_weight + '_' + str(iter) + '.pth')
        torch.save(self.critic.state_dict(), dir_path + critic_weight + '_' + str(iter) + '.pth')
        torch.save(self.V_net.state_dict(), dir_path + v_net_weight + '_' + str(iter) + '.pth')


    def load_model(self, dir_path, actor_weight, critic_weight, v_net_weight, iter):
        self.actor.load_state_dict(torch.load(dir_path + actor_weight + '_' + str(iter) + '.pth'))
        self.critic.load_state_dict(torch.load(dir_path + critic_weight + '_' + str(iter) + '.pth'))
        self.V_net.load_state_dict(torch.load(dir_path + v_net_weight + '_' + str(iter) + '.pth'))


    def train(self, obs, obs_topic, actions, pi, pi_old, reward, next_obs, next_obs_topic, target_net_flag):
        #  obs.shape = (num_client, num_topic, channel=9, 81, 81)
        #  obs_topic.shape = (num_topic, channel=3)
        #  actions.shaoe = (num_topic, num_client)
        #  pi.shape = (num_topic, num_client, N_action)
        #  next_obs.shape = (num_client, num_topic, channel=9, 81, 81)
        #  next_obs_topic = (num_topic, channel=3)

        # 行動のtensor化
        actions_onehot = torch.zeros(self.num_topic*self.num_agent*self.N_action, device=self.device)

        for t in range(self.num_topic):
            for i in range(self.num_agent):
                action = int(actions[t][i])
                if action != -1:
                    actions_onehot[t*self.num_agent*self.N_action + i*self.N_action + action] = 1
                else:
                    idx = t*self.num_agent*self.N_action + i*self.N_action
                    actions_onehot[idx:idx+self.N_action].fill_(-1)
        
        #  ==========経験再生用バッファへの追加==========
        obs_tensor = torch.FloatTensor(obs[0]).to(self.device)
        state_topic = torch.FloatTensor(obs_topic).to(self.device).view(-1)
        next_obs_tensor = torch.FloatTensor(next_obs[0]).to(self.device)
        next_state_topic = torch.FloatTensor(next_obs_topic).to(self.device).view(-1)

        publisher_distribution = obs_tensor[:, 1]
        subscriber_distribution = obs_tensor[:, 2]
        topic_storage_info = obs_tensor[:, 4]
        topic_cpu_info = obs_tensor[:, 7]

        next_publisher_distribution = next_obs_tensor[:, 1]
        next_subscriber_distribution = next_obs_tensor[:, 2]
        next_topic_storage_info = next_obs_tensor[:, 4]
        next_topic_cpu_info = next_obs_tensor[:, 7]

        state = torch.zeros((self.num_topic*4 + 2, 81, 81), device=self.device)
        next_state = torch.zeros((self.num_topic*4 + 2, 81, 81), device=self.device)

        state[:self.num_topic] = publisher_distribution
        state[self.num_topic:2*self.num_topic] = subscriber_distribution
        state[2*self.num_topic:3*self.num_topic] = topic_storage_info
        state[3*self.num_topic:4*self.num_topic] = topic_cpu_info
        state[-2:] = obs_tensor[0][5:7]

        next_state[:self.num_topic] = next_publisher_distribution
        next_state[self.num_topic:2*self.num_topic] = next_subscriber_distribution
        next_state[2*self.num_topic:3*self.num_topic] = next_topic_storage_info
        next_state[3*self.num_topic:4*self.num_topic] = next_topic_cpu_info
        next_state[-2:] = next_obs_tensor[0][5:7]

        self.replay_buffer.add(state, state_topic, actions_onehot, reward, next_state, next_state_topic)

        if len(self.replay_buffer) < self.buffer_size:
            #print(f"replay buffer < buffer size ({len(self.replay_buffer)})")
            return
        
        obs_exp, obs_topic_exp, actions_onehot_exp, reward_exp, next_obs_exp, next_obs_topic_exp = self.replay_buffer.get_batch()

        reward_exp = torch.FloatTensor(reward_exp).unsqueeze(1).to(self.device)

        if target_net_flag:
            self.target_V_net.load_state_dict(self.V_net.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())

        #  ========== V_net の更新 ==========

        V_target = reward_exp/100 + self.gamma*self.target_V_net.get_value(next_obs_exp, next_obs_topic_exp)
        V = self.V_net.get_value(obs_exp, obs_topic_exp)

        V_net_loss = self.V_net_loff_fn(V_target.detach(), V)

        self.V_net_optimizer.zero_grad()
        V_net_loss.backward(retain_graph=True)
        self.V_net_optimizer.step()

        Q1 = self.critic.get_value(obs_exp, obs_topic_exp, actions_onehot_exp)

        #  ========== critic ネットワークの更新 ==========
        critic_loss = self.critic_loss_fn(V_target.detach(), Q1)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        critic_obs = torch.stack([state] * self.num_agent, dim=0)
        critic_obs_topic = torch.stack([state_topic] * self.num_agent, dim=0)
        critic_action = torch.stack([actions_onehot] * self.num_agent, dim=0)
                    
        Q2 = self.critic.get_value(critic_obs, critic_obs_topic, critic_action)

        Q_tmp = torch.zeros(self.num_topic, self.num_agent, self.N_action, device=self.device)

        critic_obs_tmp = torch.stack([state] * self.num_topic*self.N_action*self.num_agent, dim=0)
        critic_obs_topic_tmp = torch.stack([state_topic] * self.num_topic*self.N_action*self.num_agent, dim=0)
        critic_action_tmp = torch.stack([actions_onehot] * self.num_topic*self.N_action*self.num_agent, dim=0)

        #  ========== actor ネットワークの更新 ==========
        #  ========== アドバンテージの計算 ==========
        E = torch.eye(self.N_action).to(self.device)
        for t in range(self.num_topic):
            actions_t = actions[t]
            mask = (actions_t != -1)
            idx_base = t * self.num_agent * self.N_action
            for i in range(self.num_agent):
                idx = idx_base + i * self.N_action
                for a in range(self.N_action):
                    if mask[i]:
                        critic_action_tmp[t*self.N_action*self.num_agent + i*self.N_action + a][idx:idx+self.N_action] = E[a]
      
        Q = self.target_critic.get_value(critic_obs_tmp, critic_obs_topic_tmp, critic_action_tmp).squeeze(1)

        Q_tmp = Q.view(self.num_topic, self.num_agent, -1)
            
        A = Q2.squeeze(1).unsqueeze(0).repeat(3,1) - torch.sum(pi * Q_tmp, dim=2)

        #  ========== ratio の計算 ==========
        mask = actions != -1
        rations = pi[mask, actions[mask]].detach() / (pi_old[mask, actions[mask]].detach() + 1e-16)

        surr1 = rations * A[mask]
        surr2 = torch.clamp(rations, 1-self.eps_clip, 1+self.eps_clip) * A[mask]

        clip = torch.min(surr1, surr2)

        #  ========== actor_loss の計算 ==========     

        actor_loss = -(clip * torch.log(pi[mask, actions[mask]] + 1e-16)).sum() / mask.sum()
        #actor_loss_old = -(A[mask] * torch.log(pi[mask, actions[mask]] + 1e-16)).sum() / mask.sum()
        #test = - torch.log(pi[mask, actions[mask]] + 1e-16).sum() / mask.sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        