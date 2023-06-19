from torch.distributions.categorical import Categorical
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, N_actions):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.N_actions = N_actions

    
    def add(self, data):
        self.buffer.append(data)

    
    def __len__(self):
        return len(self.buffer)


    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        
        actor_obs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0)
        actor_obs_topic = torch.cat([x[1].unsqueeze(0) for x in data], dim=0)
        critic_obs = torch.cat([x[2].unsqueeze(0) for x in data], dim=0)
        critic_obs_topic = torch.cat([x[3].unsqueeze(0) for x in data], dim=0)
        next_actor_obs = torch.cat([x[4].unsqueeze(0) for x in data], dim=0)
        next_actor_obs_topic = torch.cat([x[5].unsqueeze(0) for x in data], dim=0)
        next_critic_obs = torch.cat([x[6].unsqueeze(0) for x in data], dim=0)
        next_critic_obs_topic = torch.cat([x[7].unsqueeze(0) for x in data], dim=0)
        actions = torch.cat([x[8].unsqueeze(0) for x in data], dim=0)
        pi = torch.cat([x[9].unsqueeze(0) for x in data], dim=0)
        reward = torch.cat([x[10] for x in data])
        advantage = torch.cat([x[11] for x in data], dim=0)

        return actor_obs, actor_obs_topic, critic_obs, critic_obs_topic, next_actor_obs, next_actor_obs_topic, next_critic_obs, next_critic_obs_topic, actions, pi, reward, advantage
    

    def reset(self):
        self.buffer.clear()


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

        self.pool1 = nn.MaxPool2d(3)
        self.pool2 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(4*9*9 + 3, 126)
        self.fc2 = nn.Linear(126, 64)
        self.fc3 = nn.Linear(64, self.N_action)
    

    def get_action(self, obs, obs_topic):
        out = self.batch_norm2d_1(obs)
        out = F.relu(self.batch_norm2d_2(self.conv1(out)))
        out = F.relu(self.batch_norm2d_3(self.conv2(out)))
        out = self.pool1(out)
        out = F.relu(self.batch_norm2d_4(self.conv3(out)))
        out = self.pool2(out)
        out = out.view(-1, 4*9*9)

        out1 = self.batch_norm1d_1(obs_topic)
        out = torch.cat([out, out1], 1)
        out = F.relu(self.batch_norm1d_2(self.fc1(out)))
        out = F.relu(self.batch_norm1d_3(self.fc2(out)))
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        
        return out


class Critic(nn.Module):

    def __init__(self, num_client, num_topic):
        super(Critic, self).__init__()
        self.N_client = num_client
        self.N_topic = num_topic

        self.batch_norm2d_1 = nn.BatchNorm2d(self.N_topic*4 + 2)
        self.batch_norm2d_2 = nn.BatchNorm2d(self.N_topic*4 + 2)
        self.batch_norm2d_3 = nn.BatchNorm2d(4)
        self.batch_norm2d_4 = nn.BatchNorm2d(2)
        self.batch_norm_topic = nn.BatchNorm1d(3*self.N_topic)
        self.batch_norm1d_1 = nn.BatchNorm1d(128)
        self.batch_norm1d_2 = nn.BatchNorm1d(64)

        self.conv1 = nn.Conv2d(in_channels=self.N_topic*4+2, out_channels=self.N_topic*4+2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.N_topic*4+2, out_channels=4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(3)
        self.pool2 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(2*9*9+3*self.N_topic, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        

    def get_value(self, S, S_topic):
        out = self.conv1(self.batch_norm2d_1(S))
        out = self.pool1(F.relu(self.batch_norm2d_2(out)))
        out = self.pool2(F.relu(self.batch_norm2d_3(self.conv2(out))))
        out = F.relu(self.batch_norm2d_4(self.conv3(out)))
        out = out.view(-1, 2*9*9)

        out_topic = self.batch_norm_topic(S_topic)
        
        out = torch.cat([out, out_topic], 1)
        out = F.relu(self.batch_norm1d_1(self.fc1(out)))
        out = F.relu(self.batch_norm1d_2(self.fc2(out)))
        out = self.fc3(out)

        return out


class HAPPO:
    
    def __init__(self, N_action, num_agent, num_topic, buffer_size, batch_size, episord_len, eps_clip, device):
        self.N_action = N_action
        self.num_agent = num_agent
        self.num_topic = num_topic
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.episord_len = episord_len
        self.eps_clip = eps_clip
        self.device = device
        self.actor = Actor(self.N_action)
        self.critic = Critic(self.N_action, num_topic)
        self.target_critic = Critic(self.N_action, num_topic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, N_actions=self.N_action)
        self.tmp_buffer = []
        self.train_iter = 0

        # オプティマイザーの設定
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        if self.device != 'cpu':
            self.actor.cuda(self.device)
            self.critic.cuda(self.device)
            self.target_critic.cuda(self.device)

        self.gamma = 0.95
        self.lamda = 0.95
        self.critic_loss_fn = torch.nn.MSELoss()


    def get_acction(self, obs, obs_topic, env, train_flag, pretrain_flag):        
        obs = obs.reshape(-1, 9, 81, 81).to(self.device)
        obs_topic = obs_topic.reshape(-1, 3).to(self.device)

        pi = self.actor.get_action(obs, obs_topic).reshape(self.num_topic, self.num_agent, self.N_action).to('cpu')

        actions = torch.full((self.num_topic, self.num_agent), -1)
        
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

        return actions, pi
    
    
    def save_model(self, dir_path, actor_weight, critic_weight, iter):
        torch.save(self.actor.state_dict(), dir_path + actor_weight + '_' + str(iter) + '.pth')
        torch.save(self.critic.state_dict(), dir_path + critic_weight + '_' + str(iter) + '.pth')


    def load_model(self, dir_path, actor_weight, critic_weight, iter):
        self.actor.load_state_dict(torch.load(dir_path + actor_weight + '_' + str(iter) + '.pth'))
        self.critic.load_state_dict(torch.load(dir_path + critic_weight + '_' + str(iter) + '.pth'))

    
    def process_input(self, obs, obs_topic):
        #  状態の tensor 化
        obs_tensor = torch.FloatTensor(obs)
        obs_topic_tensor = torch.FloatTensor(obs_topic)

        #  actor network 用の input の作成
        actor_obs = obs_tensor.permute(1, 0, 2, 3, 4)

        obs_topic_channel = 3
        actor_obs_topic = torch.zeros((self.num_topic, self.num_agent, obs_topic_channel))

        for t in range(self.num_topic):
            for agent_id in range(self.num_agent):
                actor_obs_topic[t][agent_id] = obs_topic_tensor[t]

        #  critic network 用の input の作成
        critic_obs_topic = obs_topic_tensor.view(-1)

        obs_tensor_tmp = obs_tensor[0]

        publisher_distribution = obs_tensor_tmp[:, 1]
        subscriber_distribution = obs_tensor_tmp[:, 2]
        topic_storage_info = obs_tensor_tmp[:, 4]
        topic_cpu_info = obs_tensor_tmp[:, 7]

        critic_obs = torch.zeros((self.num_topic*4 + 2, 81, 81))

        critic_obs[:self.num_topic] = publisher_distribution
        critic_obs[self.num_topic:2*self.num_topic] = subscriber_distribution
        critic_obs[2*self.num_topic:3*self.num_topic] = topic_storage_info
        critic_obs[3*self.num_topic:4*self.num_topic] = topic_cpu_info
        critic_obs[-2:] = obs_tensor_tmp[0][5:7]

        return actor_obs, actor_obs_topic, critic_obs, critic_obs_topic

    
    def collect(self, actor_obs, actor_obs_topic, critic_obs, critic_obs_topic, next_actor_obs, next_actor_obs_topic, next_critic_obs, next_critic_obs_topic, actions, pi, reward):
        #  ==========経験再生用バッファへの追加==========
        advantage = 0

        reward = reward / 100 + 5

        reward = torch.FloatTensor([reward])

        self.tmp_buffer.append([actor_obs, actor_obs_topic, critic_obs, critic_obs_topic, next_actor_obs, next_actor_obs_topic, next_critic_obs, next_critic_obs_topic, actions, pi.detach(), reward, advantage])

    
    def compute_advantage(self):
    
        value = torch.zeros((self.episord_len))
        value_target = torch.zeros((self.episord_len))

        for time in range(self.episord_len):
            data = self.tmp_buffer[time]

            state = data[2].unsqueeze(0).to(self.device)
            state_topic = data[3].unsqueeze(0).to(self.device)

            state = torch.cat([state, state], dim=0)
            state_topic = torch.cat([state_topic, state_topic], dim=0)

            value[time] = self.critic.get_value(state, state_topic)[0]
            value_target[time] = self.target_critic.get_value(state, state_topic)[0]

        value = value.detach()
        value_target = value_target.detach()

        data = self.tmp_buffer[-1]
        reward = data[-2]
        gae = reward - value[-1]
        data[-1] = gae
        
        for time in reversed(range(self.episord_len-1)):
            data = self.tmp_buffer[time]
            reward = data[-2]

            delta = reward + self.gamma*value_target[time+1] - value[time]
            gae = delta + self.lamda*self.gamma*gae

            data[-1] = gae

        for data in self.tmp_buffer:
            self.replay_buffer.add(tuple(data))

        self.tmp_buffer = []
    

    def train(self, target_net_iter):

        if len(self.replay_buffer) < self.buffer_size:
            print(f"replay buffer < buffer size ({len(self.replay_buffer)})")
            return
        
        if self.train_iter == target_net_iter:
            print(f"critic_target update")
            self.train_iter = 0
            self.target_critic.load_state_dict(self.critic.state_dict())
        
        actor_obs_exp, actor_obs_topic_exp, critic_obs_exp, critic_obs_topic_exp, next_actor_obs_exp, next_actor_obs_topic_exp, next_critic_obs_exp, next_critic_obs_topic_exp, actions_exp, pi_exp, reward_exp, advantage_exp = self.replay_buffer.get_batch()
        #  actor_obs_exp = torch.Size([batch_size * episode_len, num_topic, num_agent, channel=9, obs_size=81, obs_size=81]), cpu
        #  actor_obs_topic_exp = torch.Size([batch_size * episode_len, num_topic, num_agent, channel=3]), cpu
        #  critic_obs_exp = torch.Size([batch_size * episode_len, channel=14, 81, 81]), cpu
        #  critic_obs_topic_exp = torch.Size([batch_size * episode_len, num_topic * channel=3]), cpu
        #  actions_exp = torch.Size([batch_size * episode_len, num_topic, num_agent]), cpu
        #  pi_exp = torch.Size([batch_size * episode_len, num_topic, num_agent, N_action]), cpu
        #  reward_exp = torch.Size(batch_size * episode_len), cpu
        #  advantage_exp = torch.Size([batch_size * episode_len]), cpu

        actor_obs_exp = actor_obs_exp.to(self.device)
        actor_obs_topic_exp = actor_obs_topic_exp.to(self.device)
        critic_obs_exp = critic_obs_exp.to(self.device)
        critic_obs_topic_exp = critic_obs_topic_exp.to(self.device)
        next_actor_obs_exp = next_actor_obs_exp.to(self.device)
        next_actor_obs_topic_exp = next_actor_obs_topic_exp.to(self.device)
        next_critic_obs_exp = next_critic_obs_exp.to(self.device)
        next_critic_obs_topic_exp = next_critic_obs_topic_exp.to(self.device)
        pi_exp = pi_exp.to(self.device)
        reward_exp = reward_exp.to(self.device).unsqueeze(1)
        advantage_exp = advantage_exp.to(self.device)

        actions_exp = actions_exp.reshape(self.batch_size, self.num_topic, self.num_agent)
        pi_exp = pi_exp.reshape(self.batch_size, self.num_topic, self.num_agent, self.N_action)

        actions_exp_onehot = torch.zeros((self.batch_size, self.num_topic, self.num_agent, self.N_action))

        for batch in range(self.batch_size):
            for topic in range(self.num_topic):
                for agent_id in range(self.num_agent):
                    action = actions_exp[batch][topic][agent_id]

                    if action != -1:
                        actions_exp_onehot[batch][topic][agent_id][action] = 1                                

        #  ========== critic ネットワークの更新 ==========
        Q1 = self.critic.get_value(critic_obs_exp, critic_obs_topic_exp)
        Q_target = reward_exp + self.gamma*self.target_critic.get_value(next_critic_obs_exp, next_critic_obs_topic_exp)
        critic_loss = self.critic_loss_fn(Q_target, Q1)

        print(f"critic_loss = {critic_loss}")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #  ========== actor ネットワークの更新 ==========
        #  ========== アドバンテージの計算 ==========
        M = advantage_exp.clone().detach()
        agent_perm = torch.randperm(self.num_agent)
        topic_perm = torch.randperm(self.num_topic)

        for agent_id in agent_perm:
            for topic_id in topic_perm:
                actor_input1 = actor_obs_exp[:, topic_id, agent_id].reshape(-1, 9, 81, 81)
                actor_input2 = actor_obs_topic_exp[:, topic_id,agent_id].reshape(-1, 3)
                mask = actions_exp_onehot[:, topic_id, agent_id].bool()

                pi_new = self.actor.get_action(actor_input1.detach(), actor_input2.detach()).reshape(self.batch_size, self.N_action)

                rations = torch.exp(torch.log(pi_new[mask] + 1e-16) - torch.log(pi_exp[:, topic_id, agent_id][mask] + 1e-16))

                rations_size = int(rations.numel())

                if rations_size != 0:
                    print(f"M = {M}")
                    surr1 = rations * M
                    surr2 = torch.clamp(rations, 1-self.eps_clip, 1+self.eps_clip) * M


                    actor_loss = torch.mean(torch.min(surr1, surr2))

                    entropy = - torch.sum(pi_exp[:, topic_id, agent_id][mask] * torch.log(pi_exp[:, topic_id, agent_id][mask] + 1e-16))

                    loss = - (actor_loss + 0.01*entropy)

                    print(f"actor_loss = {actor_loss}, entropy = {entropy}")

                    self.actor_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.actor_optimizer.step()

                    pi_new_update = self.actor.get_action(actor_input1.detach(), actor_input2.detach()).reshape(self.batch_size, self.N_action)

                    M = (torch.exp(torch.log(pi_new_update[mask] + 1e-16) - torch.log(pi_new[mask] + 1e-16)) * M).detach()

        self.replay_buffer.reset()
        self.train_iter += 1
        