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
    def __init__(self, device, max_agent, max_topic, data_index_path, output_file):

        self.device = device
        self.max_agent = max_agent
        self.max_topic = max_topic
        self.batch_size = 1

        self.obs_size = 27

        self.random_flag = True

        self.output_file = output_file

        with open(output_file, "w") as f:
            pass

        #  環境のインスタンスの生成
        self.data_index_path = data_index_path

        df_index = pd.read_csv(data_index_path, index_col=0, dtype=str)
        df_index.at['data', 'solve_file'] = output_file
        df_index.to_csv(data_index_path)

        self.env = Env(data_index_path)
        self.num_agent = self.env.num_client
        self.num_topic = self.env.num_topic
        agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
        obs, mask = self.env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
        self.obs_dim = obs[0][0].shape[0]

        if self.env.simulation_time % self.env.time_step == 0:
            self.episode_length = int(self.env.simulation_time / self.env.time_step)
        else:
            sys.exit("simulation_time が time_step の整数倍になっていません")

        # 各種パラメーター
        self.N_action = 9

        self.obs_distri_dim = self.obs_size*self.obs_size
        self.obs_info_dim = self.obs_dim - self.obs_distri_dim*9

        self.policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.obs_info_dim, self.N_action, self.batch_size, self.num_agent, self.num_topic, self.max_agent, self.max_topic, device=self.device, multi=False)

        self.trainer = MATTrainer(self.policy, self.num_agent, self.device)

        self.buffer = SharedReplayBuffer(self.episode_length, self.batch_size, self.num_agent, self.num_topic, self.obs_dim, self.N_action)

        self.batch = 0

        
    def warmup(self, env, batch, train=True):
        env.reset()

        agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
        obs, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)

        if train:
            self.buffer.obs[0][batch] = obs.reshape(self.num_agent*self.num_topic, self.obs_dim).copy()
            self.buffer.mask[0][batch] = np.bool_(mask.reshape(self.num_agent*self.num_topic).copy())

            self.buffer.agent_perm[0][batch] = agent_perm
            self.buffer.topic_perm[0][batch] = topic_perm

        else:
            self.test_buffer.obs[0][batch] = obs.reshape(self.num_agent*self.num_topic, self.obs_dim).copy()
            self.test_buffer.mask[0][batch] = np.bool_(mask.reshape(self.num_agent*self.num_topic).copy())

            self.test_buffer.agent_perm[0][batch] = agent_perm
            self.test_buffer.topic_perm[0][batch] = topic_perm

    
    @torch.no_grad()
    def collect_batch(self, step, train=True):
        #  TransformerPolicy を学習用に設定
        self.trainer.prep_rollout()

        if train:
            # mask.shape = (16, 90)
            # action_distribution.shape = torch.Size([16, 90, 9])
            value, action, action_log_prob = self.trainer.policy.get_actions(self.buffer.obs[step], self.buffer.mask[step])
        else:
            value, action, action_log_prob = self.trainer.policy.get_actions(self.test_buffer.obs[step], self.test_buffer.mask[step], deterministic=True)

        #  _t2n: tensor → numpy
        values = np.array(_t2n(value))
        actions = np.array(_t2n(action))
        action_log_probs = np.array(_t2n(action_log_prob))

        return values, actions, action_log_probs
    

    def insert_batch(self, obs, mask, rewards, values, actions, action_log_probs, agent_perm, topic_perm, train=True):

        if train:
            self.buffer.insert_batch(obs, mask, actions, action_log_probs, values, rewards, agent_perm, topic_perm)
        else:
            self.test_buffer.insert_batch(obs, mask, actions, action_log_probs, values, rewards, agent_perm, topic_perm)

        
    def get_perm(self, random_flag=True):
        
        agent_list = range(self.num_agent)
        topic_list = range(self.num_topic)

        if random_flag:
            agent_perm = random.sample(agent_list, self.num_agent)
            topic_perm = random.sample(topic_list, self.num_topic)
        else:
            agent_perm = list(agent_list)
            topic_perm = list(topic_list)

        return agent_perm, topic_perm
    

    def execution(self, load_parameter_path):

        self.policy.restore(load_parameter_path)
            
        #  環境のリセット
        self.warmup(self.env, 0)

        #  1エピソード中の reward の保持
        reward_history = []        

        #  各エピソードにおける時間の推移
        for time in range(0, self.env.simulation_time, self.env.time_step):
            #  print(f"batch, epi_iter, time = {self.batch}, {epi_iter}, {time}")

            step = int(time / self.env.time_step)

            values_batch, actions_batch, action_log_probs_batch = self.collect_batch(step)

            # 報酬の受け取り
            reward_batch = np.zeros((self.batch_size), dtype=np.float32)

            agent_perm_batch = np.zeros((self.batch_size, self.num_agent), dtype=np.int64)
            topic_perm_batch = np.zeros((self.batch_size, self.num_topic), dtype=np.int64)

            obs_batch = np.zeros((self.batch_size, self.num_agent, self.num_topic, self.obs_dim), dtype=np.float32)
            mask_batch = np.zeros((self.batch_size, self.num_agent, self.num_topic), dtype=bool)

            reward = self.env.step(actions_batch[0], self.buffer.agent_perm[step][0], self.buffer.topic_perm[step][0], time)
            reward_history.append(reward)
            reward_batch[0] = -reward

            for i in range(self.num_agent):
                client = self.env.pre_time_clients[i]
                util.write_solution_csv(self.output_file, DataSolution(client.id, time, client.x, client.y, client.pub_edge, client.sub_edge), self.num_topic)

            #  状態の観測
            #  ランダムな順にいつか改修
            agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
            agent_perm_batch[0] = agent_perm
            topic_perm_batch[0] = topic_perm

            obs, mask = self.env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
            obs_batch[0] = obs
            mask_batch[0] = mask

            self.insert_batch(obs_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)

        print(f"reward = {sum(reward_history)}")


device = "cuda:1"
data_index_path = "../dataset/debug/debug/index/index_easy.csv"
output_file = "../dataset/debug/debug/solution/easy_test.csv"
output_animation_file = "../dataset/debug/debug/animation/easy_execution_animation.gif"
load_parameter_path = '../result/temporary/debug/easy/model_parameter/transformer_easy_mat_batch_long0_5000.pth'

max_agent = 30
max_topic = 3

executer = MATExecuter(device, max_agent, max_topic, data_index_path, output_file)

executer.execution(load_parameter_path)

animation.create_assign_animation(data_index_path, output_animation_file, FPS=5)
#animation.create_single_assign_animation(data_index_path, output_animation_file, FPS=5)