from env import Env
from MAT.ma_transformer import MultiAgentTransformer
from MAT.transformer_policy import TransformerPolicy
from MAT.mat_trainer import MATTrainer
from MAT.shared_buffer import SharedReplayBuffer
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import random
import torch
import numpy as np

sys.path.append("../../dataset_visualization/src")
import util
import animation
from data import DataSolution


def _t2n(x):
    return x.detach().cpu().numpy()


class MATExecuter:
    def __init__(self, device, data_index_path=None):

        self.device = device
        self.batch_size = 1

        self.obs_size = 27

        #  環境のインスタンスの生成
        if data_index_path is not None:
            self.data_index_path = data_index_path
            self.env = Env(data_index_path)
            self.num_agent = self.env.num_client
            self.num_topic = self.env.num_topic
            agent_perm, topic_perm = self.random_perm()
            obs, mask = self.env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
            self.obs_dim = obs[0][0].shape[0]
            if self.env.simulation_time % self.env.time_step == 0:
                self.episode_length = int(self.env.simulation_time / self.env.time_step)
            else:
                sys.exit("simulation_time が time_step の整数倍になっていません")
        else:
            sys.exit("使用するデータを指定して下さい")

        # 各種パラメーター
        self.N_action = 9

        self.obs_distri_dim = self.obs_size*self.obs_size
        self.obs_info_dim = self.obs_dim - self.obs_distri_dim*9

        self.policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.obs_info_dim, self.N_action, self.batch_size, self.num_agent, self.num_topic, self.device)

        self.trainer = MATTrainer(self.policy, self.num_agent, self.device)

        self.buffer = SharedReplayBuffer(self.episode_length, self.batch_size, self.num_agent, self.num_topic, self.obs_dim, self.N_action)

        self.batch = 0

        
    def warmup(self, agent_perm, topic_perm):
        obs, mask = self.env.get_observation_mat(agent_perm, topic_perm, self.obs_size)

        self.buffer.obs[0][self.batch] = obs.reshape(self.num_agent*self.num_topic, self.obs_dim).copy()
        self.buffer.mask[0][self.batch] = np.bool_(mask.reshape(self.num_agent*self.num_topic).copy())

        self.buffer.agent_perm[0][self.batch]
        self.buffer.topic_perm[0][self.batch]

    
    @torch.no_grad()
    def collect(self, batch, step, near_action):
        #  TransformerPolicy を学習用に設定
        self.trainer.prep_rollout()

        action_distribution = torch.ones((1, self.num_agent*self.num_topic, self.N_action))*-1

        value, action, action_log_prob, action_distribution[:, self.buffer.mask[step][batch]] = self.trainer.policy.get_actions(self.buffer.obs[step][batch], self.buffer.mask[step][batch], near_action)

        #  _t2n: tensor → numpy
        values = np.array(_t2n(value))
        actions = np.array(_t2n(action))
        action_log_probs = np.array(_t2n(action_log_prob))
        action_distribution = np.array(_t2n(action_distribution))

        return values, actions, action_log_probs, action_distribution
    

    def insert(self, batch, obs, mask, rewards, values, actions, action_log_probs, agent_perm, topic_perm):

        self.buffer.insert(batch, obs, mask, actions, action_log_probs, values, rewards, agent_perm, topic_perm)

        
    def random_perm(self):
        
        agent_list = range(self.num_agent)
        topic_list = range(self.num_topic)

        agent_perm = random.sample(agent_list, self.num_agent)
        topic_perm = random.sample(topic_list, self.num_topic)

        return agent_perm, topic_perm
    

    def execution(self, output, load_parameter_path):

        self.policy.restore(load_parameter_path)
            
        #  環境のリセット
        self.env.reset()

        agent_perm, topic_perm = self.random_perm()

        self.warmup(agent_perm, topic_perm)

        #  1エピソード中の reward の保持
        reward_history = []        

        #  各エピソードにおける時間の推移
        for time in range(0, self.env.simulation_time, self.env.time_step):
            #  print(f"batch, epi_iter, time = {self.batch}, {epi_iter}, {time}")

            step = int(time / self.env.time_step)

            values, actions, action_log_probs, action_distribution = self.collect(self.batch, step, near_action=None)

            # 報酬の受け取り
            reward = self.env.step(actions, agent_perm, topic_perm, time)
            reward_history.append(reward)
            reward = -reward

            for i in range(self.num_agent):
                client = self.env.clients[i]
                util.write_solution_csv(output, DataSolution(client.id, time, client.x, client.y, client.pub_edge, client.sub_edge), self.num_topic)

            #  状態の観測
            #  ランダムな順にいつか改修
            agent_perm, topic_perm = self.random_perm()

            obs, mask = self.env.get_observation_mat(agent_perm, topic_perm, self.obs_size)

            self.insert(self.batch, obs, mask, reward, values, actions, action_log_probs, agent_perm, topic_perm)


        print(f"reward = {sum(reward_history)}")


device = "cuda:1"
data_index_path = "../dataset/debug/index/index_onetopic.csv"
output_file = "../dataset/debug/solution/onetopic_test.csv"
load_parameter_path = '../result/temporary/debug/model_parameter/transformer_onetopic_mat0_10000.pth'

executer = MATExecuter(device, data_index_path)

executer.execution(output_file, load_parameter_path)