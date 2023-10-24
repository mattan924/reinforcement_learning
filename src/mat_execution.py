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

        if self.env.simulation_time % self.env.time_step == 0:
            self.episode_length = int(self.env.simulation_time / self.env.time_step)
        else:
            sys.exit("simulation_time が time_step の整数倍になっていません")

        # 各種パラメーター
        self.N_action = 9

        self.obs_dim = self.obs_size*self.obs_size*9 + 3
        self.obs_distri_dim = self.obs_size*self.obs_size
        self.obs_info_dim = self.obs_dim - self.obs_distri_dim*9

        self.policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, device=self.device, multi=True)

        self.trainer = MATTrainer(self.policy, self.device)

        self.buffer = SharedReplayBuffer(self.episode_length, self.batch_size, self.max_agent, self.max_topic, self.obs_dim, self.N_action)

        self.batch = 0

        
        #  初期準備
    def warmup(self, env, batch):
        env.reset()

        agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
        obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)

        self.buffer.obs_posi[0][batch] = obs_posi
        self.buffer.obs_publisher[0][batch] = obs_publisher
        self.buffer.obs_subscriber[0][batch] = obs_subscriber
        self.buffer.obs_distribution[0][batch] = obs_distribution
        self.buffer.obs_topic_used_storage[0][batch] = obs_topic_used_storage
        self.buffer.obs_storage[0][batch] = obs_storage
        self.buffer.obs_cpu_cycle[0][batch] = obs_cpu_cycle
        self.buffer.obs_topic_num_used[0][batch] = obs_topic_num_used
        self.buffer.obs_num_used[0][batch] = obs_num_used
        self.buffer.obs_topic_info[0][batch] = obs_topic_info
        self.buffer.mask[0][batch] = np.bool_(mask.reshape(self.max_agent*self.max_topic))

        self.buffer.agent_perm[0][batch] = agent_perm
        self.buffer.topic_perm[0][batch] = topic_perm


    @torch.no_grad()
    def collect(self, step):
        #  TransformerPolicy を学習用に設定
        self.trainer.prep_rollout()

        obs_posi = self.buffer.obs_posi[step]
        obs_client = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim*3), dtype=np.float32)
        obs_client[:, :, :self.obs_distri_dim] = self.buffer.obs_publisher[step]
        obs_client[:, :, self.obs_distri_dim:self.obs_distri_dim*2] = self.buffer.obs_subscriber[step]
        obs_client[:, :, self.obs_distri_dim*2:self.obs_distri_dim*3] = self.buffer.obs_distribution[step][:, np.newaxis]
            
        obs_edge = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim*5), dtype=np.float32)
        obs_edge[:, :, :self.obs_distri_dim] = self.buffer.obs_topic_used_storage[step]
        obs_edge[:, :, self.obs_distri_dim:self.obs_distri_dim*2] = self.buffer.obs_storage[step][:, np.newaxis]
        obs_edge[:, :, self.obs_distri_dim*2:self.obs_distri_dim*3] = self.buffer.obs_cpu_cycle[step][:, np.newaxis]
        obs_edge[:, :, self.obs_distri_dim*3:self.obs_distri_dim*4] = self.buffer.obs_topic_num_used[step]
        obs_edge[:, :, self.obs_distri_dim*4:self.obs_distri_dim*5] = self.buffer.obs_num_used[step][:, np.newaxis]
            
        obs_topic_info = self.buffer.obs_topic_info[step]

        value, action, action_log_prob = self.trainer.policy.get_actions(obs_posi, obs_client, obs_edge, obs_topic_info, self.buffer.mask[step], deterministic=False)

        #  _t2n: tensor → numpy
        values = np.array(_t2n(value))
        actions = np.array(_t2n(action))
        action_log_probs = np.array(_t2n(action_log_prob))

        return values, actions, action_log_probs
    

    def insert_batch(self, obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask, rewards, values, actions, action_log_probs, agent_perm, topic_perm):

        self.buffer.insert_batch(obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask, actions, action_log_probs, values, rewards, agent_perm, topic_perm)


    def get_perm(self, random_flag=True):
        
        agent_list = range(self.max_agent)
        topic_list = range(self.max_topic)

        if random_flag:
            agent_perm = random.sample(agent_list, self.max_agent)
            topic_perm = random.sample(topic_list, self.max_topic)
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

            values_batch, actions_batch, action_log_probs_batch = self.collect(step)

            # 報酬の受け取り
            reward_batch = np.zeros((self.batch_size), dtype=np.float32)

            agent_perm_batch = np.zeros((self.batch_size, self.max_agent), dtype=np.int64)
            topic_perm_batch = np.zeros((self.batch_size, self.max_topic), dtype=np.int64)

            obs_posi_batch = np.zeros((self.batch_size, self.max_agent, self.obs_distri_dim), dtype=np.float32)
            obs_publisher_batch = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_subscriber_batch = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_distribution_batch = np.zeros((self.batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_topic_used_storage_batch = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_storage_batch = np.zeros((self.batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_cpu_cycle_batch = np.zeros((self.batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_topic_num_used_batch = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_num_used_batch = np.zeros((self.batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_topic_info_batch = np.zeros((self.batch_size, self.max_topic, 3), dtype=np.float32)

            mask_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic), dtype=np.bool)

            reward = self.env.step(actions_batch[0][self.buffer.mask[step][0]], self.buffer.agent_perm[step][0], self.buffer.topic_perm[step][0], time)
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

            obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask = self.env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
            obs_posi_batch[0] = obs_posi
            obs_publisher_batch[0] = obs_publisher
            obs_subscriber_batch[0] = obs_subscriber
            obs_distribution_batch[0] = obs_distribution
            obs_topic_used_storage_batch[0] = obs_topic_used_storage
            obs_storage_batch[0] = obs_storage
            obs_cpu_cycle_batch[0] = obs_cpu_cycle
            obs_topic_num_used_batch[0] = obs_topic_num_used
            obs_num_used_batch[0] = obs_num_used
            obs_topic_info_batch[0] = obs_topic_info
            mask_batch[0] = mask

            self.insert_batch(obs_posi_batch, obs_publisher_batch, obs_subscriber_batch, obs_distribution_batch, obs_topic_used_storage_batch, obs_storage_batch, obs_cpu_cycle_batch, obs_topic_num_used_batch, obs_num_used_batch, obs_topic_info_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)

        return sum(reward_history)
 

device = "cuda:1"
data_index_path = "../dataset/debug/easy/regular_meeting/train/index/index_easy_hight_load_0.csv"
output_file = "../dataset/debug/easy/regular_meeting/train/solution/solution_easy_hight_load_0.csv"
#output_animation_file = "../dataset/debug/debug/animation/easy_execution_animation.gif"
load_parameter_path_base = '../result/temporary/regular_meeting/model_parameter/transformer_easy_hight_load_multi_batch16_epoch8_block1_1_'
log_file = "../result/temporary/regular_meeting/easy_hight_load_multi_train3.log"

with open(log_file, "w") as f:
    pass

max_agent = 30
max_topic = 3

max_iter = 3000

executer = MATExecuter(device, max_agent, max_topic, data_index_path, output_file)

for epi in range(0, max_iter+1, 1000):
    load_parameter_path = load_parameter_path_base + str(epi) + ".pth"
    
    reward = executer.execution(load_parameter_path)

    with open(log_file, "a") as f:
        f.write(f"{epi}, {reward}\n")

#animation.create_single_assign_animation(data_index_path, output_animation_file, FPS=5)
