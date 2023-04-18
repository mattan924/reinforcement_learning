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

    
    def add(self, state, state_topic, actions, actions_onehot, reward, next_state, next_state_topic):
        data = (state, state_topic, actions, actions_onehot, reward, next_state, next_state_topic)
        self.buffer.append(data)

    
    def __len__(self):
        return len(self.buffer)


    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.cat([x[0].unsqueeze(0) for x in data], dim=0)
        state_topic = torch.cat([x[1].unsqueeze(0) for x in data], dim=0)
        actions = np.stack([x[2] for x in data])
        actions_onehot = torch.cat([x[3].unsqueeze(0) for x in data], dim=0)
        reward = np.stack([x[4] for x in data])
        next_state = torch.cat([x[5].unsqueeze(0) for x in data], dim=0)
        next_state_topic = torch.cat([x[6].unsqueeze(0) for x in data], dim=0)

        return state, state_topic, actions, actions_onehot, reward, next_state, next_state_topic


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
        self.batch_norm_topic = nn.BatchNorm1d(3)
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

        self.fc4 = nn.Linear(2*9*9+3+64, 128)
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
        self.batch_norm_topic = nn.BatchNorm1d(3)
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

        self.fc1 = nn.Linear(2*9*9 + 3, 64)
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
    
    def __init__(self, N_action, num_agent, num_topic, buffer_size, batch_size, device):
        self.N_action = N_action
        self.num_agent = num_agent
        self.num_topic = num_topic
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.actor = Actor(self.N_action)
        self.critic = Critic(self.N_action, num_agent, num_topic)
        self.V_net = V_Net(num_topic)
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, N_actions=self.N_action, device=self.device)

        # オプティマイザーの設定
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.V_net_optimizer = torch.optim.Adam(self.V_net.parameters(), lr=1e-3)

        if self.device != 'cpu':
            self.actor.cuda(self.device)
            self.critic.cuda(self.device)
            self.V_net.cuda(self.device)

        self.gamma = 0.95
        self.critic_loss_fn = torch.nn.MSELoss()
        self.V_net_loff_fn = torch.nn.MSELoss()


    def get_acction(self, obs, obs_topic, env, train_flag, pretrain_flag):        
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        obs_topic_tensor = torch.FloatTensor(obs_topic).to(self.device)

        obs_tensor = torch.permute(obs_tensor, (1, 0, 2, 3, 4))

        pi = torch.zeros((self.num_topic, self.num_agent, self.N_action)).to(self.device)
        for t in range(self.num_topic):
            obs_topic_tensor_tmp = obs_topic_tensor[t].unsqueeze(0)
            for _ in range(self.num_agent-1):
                obs_topic_tensor_tmp = torch.cat([obs_topic_tensor_tmp, obs_topic_tensor[t].unsqueeze(0)], 0)
            
            pi[t] = self.actor.get_action(obs_tensor[t], obs_topic_tensor_tmp)

        actions = np.ones((self.num_topic, self.num_agent))*-1
        
        if train_flag:
            clients = env.clients
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
                for i in range(self.num_agent):
                    client = clients[i]
                    for t in range(self.num_topic):
                        if client.pub_topic[t] == 1:
                            actions[t][i] = Categorical(pi[t][i]).sample().item()
        else:
            clients = env.clients

            for i in range(self.num_agent):
                client = clients[i]
                for t in range(self.num_topic):
                    if client.pub_topic[t] == 1:
                        actions[t][i] = torch.argmax(pi[t][i])

        return actions, pi

    
    def save_model(self, dir_path, iter):
        torch.save(self.actor.state_dict(), dir_path + 'actor_weight' + '_' + str(iter) + '.pth')
        torch.save(self.critic.state_dict(), dir_path + 'critic_weight' + '_' + str(iter) + '.pth')
        torch.save(self.V_net.state_dict(), dir_path + 'v_net_weight' + '_' + str(iter) + '.pth')


    def load_model(self, dir_path, iter):
        self.actor.load_state_dict(torch.load(dir_path + 'actor_weight' + '_' + str(iter) + '.pth'))
        self.critic.load_state_dict(torch.load(dir_path + 'critic_weight' + '_' + str(iter) + '.pth'))
        self.V_net.load_state_dict(torch.load(dir_path + 'v_net_weight' + '_' + str(iter) + '.pth'))


    def train(self, obs, obs_topic, actions, pi, reward, next_obs, next_obs_topic, fix_net_flag):
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
        
        # 経験再生用バッファへの追加
        obs_tensor = torch.FloatTensor(obs[0]).to(self.device)
        state_topic = torch.FloatTensor(obs_topic).to(self.device).view(-1)
        next_obs_tensor = torch.FloatTensor(next_obs[0]).to(self.device)
        next_state_topic = torch.FloatTensor(next_obs_topic).to(self.device).view(-1)


        publisher_distribution = torch.zeros((self.num_topic, 81, 81))
        subscriber_distribution = torch.zeros((self.num_topic, 81, 81))
        topic_storage_info = torch.zeros((self.num_topic, 81, 81))
        topic_cpu_info = torch.zeros((self.num_topic, 81, 81))

        next_publisher_distribution = torch.zeros((self.num_topic, 81, 81))
        next_subscriber_distribution = torch.zeros((self.num_topic, 81, 81))
        next_topic_storage_info = torch.zeros((self.num_topic, 81, 81))
        next_topic_cpu_info = torch.zeros((self.num_topic, 81, 81))

        for t in range(self.num_topic):
            publisher_distribution[t] = obs_tensor[t][1]
            subscriber_distribution[t] = obs_tensor[t][2]
            topic_storage_info[t] = obs_tensor[t][4]
            topic_cpu_info[t] = obs_tensor[t][7]

            next_publisher_distribution[t] = next_obs_tensor[t][1]
            next_subscriber_distribution[t] = next_obs_tensor[t][2]
            next_topic_storage_info[t] = next_obs_tensor[t][4]
            next_topic_cpu_info[t] = next_obs_tensor[t][7]

        state = torch.zeros((self.num_topic*4 + 2, 81, 81), device=self.device)
        next_state = torch.zeros((self.num_topic*4 + 2, 81, 81), device=self.device)

        state[0:self.num_topic] = publisher_distribution
        state[self.num_topic:2*self.num_topic] = subscriber_distribution
        state[2*self.num_topic:3*self.num_topic] = topic_storage_info
        state[3*self.num_topic:4*self.num_topic] = topic_cpu_info
        state[-2] = obs_tensor[0][5]
        state[-1] = obs_tensor[0][6]

        next_state[0:self.num_topic] = next_publisher_distribution
        next_state[self.num_topic:2*self.num_topic] = next_subscriber_distribution
        next_state[2*self.num_topic:3*self.num_topic] = next_topic_storage_info
        next_state[3*self.num_topic:4*self.num_topic] = next_topic_cpu_info
        next_state[-2] = next_obs_tensor[0][5]
        next_state[-1] = next_obs_tensor[0][6]

        self.replay_buffer.add(state, state_topic, actions, actions_onehot, reward, next_state, next_state_topic)

        if len(self.replay_buffer) < self.buffer_size:
            print("replay_buffer size < buffer size")
            return

        obs_exp, obs_topic_exp, actions_exp, actions_onehot_exp, reward_exp, next_obs_exp, next_obs_topic_exp = self.replay_buffer.get_batch()

        reward_exp = torch.FloatTensor(reward_exp).unsqueeze(1).to(self.device)

        if fix_net_flag:
            V_target = reward_exp/100 + self.gamma*self.V_net.get_value(next_obs_exp, next_obs_topic_exp)
            V = self.V_net.get_value(obs_exp, obs_topic_exp)

            V_net_loss = self.V_net_loff_fn(V_target.detach(), V)

            self.V_net_optimizer.zero_grad()
            V_net_loss.backward(retain_graph=True)
            self.V_net_optimizer.step()

            # batch_size*1のQ値をtensorとして取得
            Q = self.critic.get_value(obs_exp, obs_topic_exp, actions_onehot_exp)

            # critic ネットワークの更新
            critic_loss = self.critic_loss_fn(V_target.detach(), Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()
        else:
            actor_loss = torch.FloatTensor([0.0]).to(self.device)

            critic_obs = state.unsqueeze(0)
            critic_obs_topic = state_topic.unsqueeze(0)
            critic_action = actions_onehot.unsqueeze(0)

            for i in range(1, self.num_agent):
                critic_obs = torch.cat([critic_obs, state.unsqueeze(0)], dim=0)
                critic_obs_topic = torch.cat([critic_obs_topic, state_topic.unsqueeze(0)], dim=0)
                critic_action = torch.cat([critic_action, actions_onehot.unsqueeze(0)], dim=0)
                    
            Q2 = self.critic.get_value(critic_obs, critic_obs_topic, critic_action)

            Q_tmp = torch.zeros(self.num_topic, self.N_action, self.num_agent, device=self.device)

            for t in range(self.num_topic):           
                for a in range(self.N_action):
                    critic_action_copy = critic_action.clone()

                    for i in range(self.num_agent):
                        for j in range(self.N_action):
                            critic_action_copy[i][t*self.num_agent*self.N_action + i*self.N_action+j] = 0
                                    
                        critic_action_copy[i][i*self.N_action + a] = 1
                            
                    Q_tmp[t][a] = self.critic.get_value(critic_obs, critic_obs_topic, critic_action_copy).squeeze(1)
                    
            Q_tmp = torch.permute(Q_tmp, (0, 2, 1))
            
            A = torch.zeros((self.num_topic, self.num_agent), device=self.device)

            for t in range(self.num_topic):
                A[t] = Q2.squeeze(1) - torch.sum(pi[t]*Q_tmp[t], 1)
                
            cnt = 0
            for t in range(self.num_topic):
                for i in range(self.num_agent):
                    if actions[t][i] != -1:
                        actor_loss = actor_loss + A[t][i].item() * torch.log(pi[t][i][int(actions[t][i])] + 1e-16)
                        cnt += 1

            actor_loss = - actor_loss / cnt

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        

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


        if self.device != 'cpu':
            self.actor.cuda(self.device)
            self.V_net.cuda(self.device)

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


    def train(self, obs, actions, pi, reward, next_obs, fix_net_flag):
       
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

        if fix_net_flag:
            V_target = reward_exp/100 + self.gamma*self.V_net.get_value(next_obs_exp)
            V = self.V_net.get_value(obs_exp)

            V_net_loss = self.V_net_loff_fn(V_target.detach(), V)

            V_net_optimizer.zero_grad()
            V_net_loss.backward(retain_graph=True)
            V_net_optimizer.step()
        else:
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
        