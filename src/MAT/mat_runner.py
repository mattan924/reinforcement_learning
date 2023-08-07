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


def read_train_curve(log_path):
    reward_history = []
    tmp = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.split(",")

            reward_history.append(float(line[1]))

            tmp += 1
            
    return reward_history


def _t2n(x):
    return x.detach().cpu().numpy()


class MATRunner:
    def __init__(self, max_epi_itr, batch_size, device, result_dir, backup_itr, learning_data_index_path=None):

        if not os.path.isdir(result_dir + "model_parameter"):
            sys.exit("結果を格納するディレクトリ" + result_dir + "model_parameter が作成されていません。")

        self.max_epi_itr = max_epi_itr
        self.batch_size = batch_size
        self.device = device
        self.result_dir = result_dir
        self.backup_itr = backup_itr

        self.obs_size = 27
        self.random_flag = False

        #  環境のインスタンスの生成
        if learning_data_index_path is not None:
            self.learning_data_index_path = learning_data_index_path
            self.env = Env(learning_data_index_path)
            self.num_agent = self.env.num_client
            self.num_topic = self.env.num_topic
            agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
            obs, mask = self.env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
            self.obs_dim = obs[0][0].shape[0]
            if self.env.simulation_time % self.env.time_step == 0:
                self.episode_length = int(self.env.simulation_time / self.env.time_step)
            else:
                sys.exit("simulation_time が time_step の整数倍になっていません")
        else:
            sys.exit("学習するデータを指定して下さい")

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

    
    @torch.no_grad()
    def compute(self, batch):
        """Calculate returns for the collected data."""
        #  transformer を評価用にセット
        self.trainer.prep_rollout()

        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.obs[-1][batch]), self.buffer.mask[-1][batch])
            
        #  _t2n: tensor から numpy への変換
        next_values = np.array(_t2n(next_values))
        
        self.buffer.compute_returns(batch, next_values, self.trainer.value_normalizer)

    
    def train(self):
        #  Transformer を train に設定
        self.trainer.prep_training()

        train_infos = self.trainer.train(self.buffer)
 
        return train_infos

    
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
    

    def train_single_env(self, output, transformer_weight, start_epi_itr, load_parameter_path=None):

        if load_parameter_path is not None:
            self.policy.restore(load_parameter_path)
            if start_epi_itr == 0:
                with open(output + ".log", 'w') as f:
                    pass
                with open(output + "_pi.log", "w") as f:
                    pass  
        else:
            with open(output + ".log", 'w') as f:
                pass

            with open(output + "_pi.log", "w") as f:
                pass   

        # 学習ループ
        for epi_iter in range(start_epi_itr, self.max_epi_itr):
            #  環境のリセット
            self.env.reset()

            agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)

            self.warmup(agent_perm, topic_perm)

            if epi_iter % 10 == 0:
                pre_train = False
            else:
                pre_train = False

            #  1エピソード中の reward の保持
            reward_history = []        

            #  各エピソードにおける時間の推移
            for time in range(0, self.env.simulation_time, self.env.time_step):
                #  print(f"batch, epi_iter, time = {self.batch}, {epi_iter}, {time}")

                step = int(time / self.env.time_step)

                #  行動と確率分布の取得
                if pre_train:
                    near_action = self.env.get_near_action(agent_perm, topic_perm)
                    # print(f"time = {time}")
                    # print(f"near_action = {near_action[:, self.buffer.mask[step][self.batch]]}")
                    values, actions, action_log_probs, action_distribution = self.collect(self.batch, step, near_action)
                    # print(f"actuin = {actions}")
                else:
                    values, actions, action_log_probs, action_distribution = self.collect(self.batch, step, near_action=None)

                # 報酬の受け取り
                reward = self.env.step(actions, agent_perm, topic_perm, time)
                reward_history.append(reward)
                reward = -reward

                #  状態の観測
                #  ランダムな順にいつか改修
                agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)

                obs, mask = self.env.get_observation_mat(agent_perm, topic_perm, self.obs_size)

                self.insert(self.batch, obs, mask, reward, values, actions, action_log_probs, agent_perm, topic_perm)

            self.compute(self.batch)

            if self.batch == self.batch_size-1:
                # 学習
                train_info = self.train()

            if epi_iter % 1 == 0:
                #  ログの出力
                #print(f"total_reward = {sum(reward_history)}")
                #print(f"train is {(epi_iter/max_epi_itr)*100}% complited.")
                with open(output + ".log", 'a') as f:
                    f.write(f"{(epi_iter/self.max_epi_itr)*100}%, {-sum(reward_history)}\n")

                with open(output + "_pi.log", "a") as f:
                    f.write(f"\n==========iter = {epi_iter} ==========\n")
                    f.write(f"agent_perm = {self.buffer.agent_perm[-2][self.batch]}\n")
                    f.write(f"topic_perm = {self.buffer.topic_perm[-2][self.batch]}\n")
                    f.write(f"agent_actions = {actions}\n")

                    action_distribution = action_distribution.reshape(-1, self.num_agent, self.num_topic, self.N_action)

                    for i in range(self.num_agent):
                        agent_idx = self.buffer.agent_perm[-2][self.batch].tolist().index(i)
                        for t in range(self.num_topic):
                            topic_idx = self.buffer.topic_perm[-2][self.batch].tolist().index(t)
                            f.write(f"agent {i} topic {t} : pi = {action_distribution[0][agent_idx][topic_idx]}\n")

            self.batch = (self.batch + 1) % self.batch_size

            #  重みパラメータのバックアップ
            if epi_iter % self.backup_itr == 0:
                self.policy.save(self.result_dir + 'model_parameter', transformer_weight, epi_iter)

        #  最適解の取り出し
        df_index = pd.read_csv(self.learning_data_index_path, index_col=0)
        opt = df_index.at['data', 'opt']

        train_curve = read_train_curve(output + ".log")

        #  学習曲線の描画
        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        wind.grid()
        wind.plot(train_curve, linewidth=1, label='COMA')
        wind.axhline(y=-opt, c='r')
        fig.savefig(output + ".png")

        #  重みパラメータの保存
        self.policy.save(self.result_dir + 'model_parameter', transformer_weight, epi_iter+1)
