from torch.distributions.categorical import Categorical
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time as time_modu


class Actor(nn.Module):
    
    def __init__(self, N_action):
        super(Actor, self).__init__()
        self.N_action = N_action
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(4*3*3, 32)
        self.fc2 = nn.Linear(32, self.N_action)
    

    def get_action(self, obs):
        out1 = self.pool1((self.conv1(obs)))
        out2 = self.pool2(torch.tanh(self.conv2(out1)))
        out3 = self.pool3(torch.tanh(self.conv3(out2)))
        out4 = out3.view(-1, 4*3*3)
        out5 = torch.tanh((self.fc1(out4)))
        out6 = F.softmax(self.fc2(out5), dim=1)
        action = Categorical(out6.squeeze(0))

        return action.sample().item(), out6


class Critic(nn.Module):

    def __init__(self, N_action, num_clinet):
        super(Critic, self).__init__()
        self.N_action = N_action
        self.N_client = num_clinet
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(num_clinet*N_action, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)

        self.fc4 = nn.Linear(4*3*3+64, 32)
        self.fc5 = nn.Linear(32, 8)
        self.fc6 = nn.Linear(8, 1)
        

    def get_value(self, S, A):
        out1_s = self.pool1(torch.tanh(self.conv1(S)))
        out2_s = self.pool2(torch.tanh(self.conv2(out1_s)))
        out3_s = self.pool3(torch.tanh(self.conv3(out2_s)))
        out4_s = out3_s.view(-1, 4*3*3)
        
        out1_a = torch.tanh(self.fc1(A))
        out2_a = torch.tanh(self.fc2(out1_a))
        out3_a = torch.tanh(self.fc3(out2_a))

        out4 = torch.cat([out4_s, out3_a], 1)
        out5 = torch.tanh(self.fc4(out4))
        out6 = torch.tanh(self.fc5(out5))
        out7 = self.fc6(out6)

        return out7


# バッチ処理未対応
class COMA:
    
    def __init__(self, N_action, num_agent, device):
        self.N_action = N_action
        self.num_agent = num_agent
        self.device = device
        self.actor = Actor(self.N_action)
        self.critic = Critic(self.N_action, num_agent)

        if self.device == 'cuda':
            self.actor.cuda()
            self.critic.cuda()

        self.gamma = 0.95
        self.alpha = 0.01
        self.critic_loss_fn = torch.nn.MSELoss()


    def get_acction(self, obs, clients):
        obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)
        
        actions = []
        pi = []
        for client in clients:
            if client.pub_topic[0] == 1:
                action, pi_k = self.actor.get_action(obs[client.id])
                actions.append(action)
                pi.append(pi_k)
            else:
                actions.append(-1)
                pi.append(0)
        
        return actions, pi


    def train(self, obs_history, actions_history, pi_history, reward_history):
        # 1エピソードの長さ
        T = len(obs_history)

        # オプティマイザーの設定
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # critic ネットワークの入力として与える状態のtensor化
        obs_critic = torch.zeros(T, 3, 81, 81, device=self.device)

        for t in range(T):
            tmp = torch.FloatTensor(obs_history[t][0][1:])
            obs_critic[t] = tmp.to(self.device)

        # critic　ネットワークの入力として与える行動のtensor化
        actions_onehot_history = torch.zeros(T, self.num_agent*self.N_action, device=self.device)
        
        for t in range(T):
            actions = actions_history[t]
            for n in range(self.num_agent):
                action = actions[n]

                if action != -1:
                    actions_onehot_history[t][n*self.N_action + action] = 1
        
        # T*1のQ値をtensorとして取得
        Q = self.critic.get_value(obs_critic, actions_onehot_history)

        # あるタイムスロット以降の報酬の合計を保存するGを初期化
        G = torch.zeros(T, 1, device=self.device)
        for time_slot in range(T):
            # reward_history からtime_slot以降の報酬を取得
            rewards = reward_history[time_slot:]

            # Gの計算
            for i in range(len(rewards)):
                G[time_slot] += pow(self.gamma, i)*rewards[i]

        # critic ネットワークの更新
        critic_loss = self.critic_loss_fn(G.detach(), Q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # 各クライアントの行動をもとに学習させる
        cnt = 0
        actor_loss = torch.FloatTensor([0.0])
        actor_loss = actor_loss.to(self.device)

        for i in range(self.num_agent):
            # アドバンテージ値のリスト
            A_list = []
            # 行動を変化させた結果を調べるために元を値を変更しないためのコピー
            actions_onehot_copy = actions_onehot_history.clone()

            for t in range(T):
                
                temp_Q = torch.zeros(1, self.N_action, device=self.device)

                if actions_history[t][i] != -1:
                    
                    for a in range(self.N_action):
                        
                        for j in range(self.N_action):
                            actions_onehot_copy[t][i*self.N_action+j] = 0
                        
                        actions_onehot_copy[t][i*self.N_action + a] = 1
                        temp_Q[0, a] = self.critic.get_value(obs_critic[t], actions_onehot_copy[t].unsqueeze(0))
                        
                    temp_A = Q[t][0] - torch.sum(pi_history[t][i]*temp_Q)
                    A_list.append(temp_A)
            
            tmp = 0
            for t in range(T): 
                if actions_history[t][i] != -1:
                    actor_loss = actor_loss + A_list[tmp].item() * torch.log(pi_history[t][i][0][actions_history[t][i]])
                    
                    tmp += 1
                    cnt += 1
            
        actor_loss = -actor_loss / cnt
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
