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


def read_train_curve(log_path, pre_train_iter):
    reward_history = []
    tmp = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.split(",")

            if tmp % pre_train_iter != 0:
                reward_history.append(float(line[1]))

            tmp += 1
            
    return reward_history


def _t2n(x):
    return x.detach().cpu().numpy()


class MATRunner:
    def __init__(self, max_epi_itr, batch_size, device, result_dir, backup_itr, learning_data_index_path=None, learning_data_index_dir=None):

        if not os.path.isdir(result_dir + "model_parameter"):
            sys.exit("結果を格納するディレクトリ" + result_dir + "model_parameter が作成されていません。")

        self.max_epi_itr = max_epi_itr
        self.batch_size = batch_size
        self.device = device
        self.result_dir = result_dir
        self.backup_itr = backup_itr

        #  環境のインスタンスの生成
        if learning_data_index_path is not None:
            self.env = Env(learning_data_index_path)
            self.num_agent = self.env.num_client
            self.num_topic = self.env.num_topic
            agent_perm, topic_perm = self.random_perm()
            self.obs_dim = self.env.get_observation_mat(agent_perm, topic_perm, obs_size=27)[0][0].shape[0]
            if self.env.simulation_time % self.env.time_step == 0:
                self.episode_length = int(self.env.simulation_time / self.env.time_step)
            else:
                sys.exit("simulation_time が time_step の整数倍になっていません")
        elif learning_data_index_path is not None:
            pass
        else:
            sys.exit("学習するデータを指定して下さい")

        # 各種パラメーター
        self.N_action = 9
        self.use_centralized_V = True
        self.use_obs_instead_of_state = False
        self.use_linear_lr_decay = False
        self.hidden_size = 64

        self.policy = TransformerPolicy(self.obs_dim, self.N_action, self.batch_size, self.num_agent, self.num_topic, self.device)

        self.trainer = MATTrainer(self.policy, self.num_agent, self.device)

        self.buffer = SharedReplayBuffer(self.episode_length, self.batch_size, self.num_agent, self.num_topic, self.obs_dim, self.N_action)

        self.batch = 0

        
    def warmup(self, agent_perm, topic_perm):
        obs = self.env.get_observation_mat(agent_perm, topic_perm, obs_size=27)

        self.buffer.obs[0][self.batch] = obs.copy()
        self.buffer.agent_perm[0][self.batch] = np.array(agent_perm)
        self.buffer.topic_perm[0][self.batch] = np.array(topic_perm)

    
    @torch.no_grad()
    def collect(self, batch, step):
        #  TransformerPolicy を学習用に設定
        self.trainer.prep_rollout()

        value, action, action_log_prob = self.trainer.policy.get_actions(self.buffer.obs[step][batch])

        #  [self.envs, agents, dim]
        #  _t2n: tensor → numpy
        #  np.split: n_rollout_threads の数に分割
        values = np.array(np.split(_t2n(value), self.batch_size))
        actions = np.array(np.split(_t2n(action), self.batch_size))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.batch_size))

        print(f"values.shape = {values.shape}")
        print(f"actions.shape = {actions.shape}")
        print(f"action_log_probs.shape = {action_log_probs.shape}")

        return values, actions, action_log_probs
    

    def insert(self, obs, rewards, values, actions, action_log_probs):

        self.buffer.insert(obs, actions, action_log_probs, values, rewards)

    
    def random_perm(self):
        agent_list = range(self.num_agent)
        topic_list = range(self.num_topic)

        agent_perm = random.sample(agent_list, self.num_agent)
        topic_perm = random.sample(topic_list, self.num_topic)

        return agent_perm, topic_perm
    

    def train_loop_single(self, output, start_epi_itr=0, load_parameter_path=None):
        if start_epi_itr != 0:
            if load_parameter_path == None:
                sys.exit("読み込む重みパラメータのパスを指定してください")
            else:
                load_flag = True
        else:
            load_flag = False

        #  学習モデルの指定
        agent = MultiAgentTransformer(self.obs_dim, self.N_action, self.batch_size, self.num_agent, self.device)

        if load_flag == True:
            agent.load_model(load_parameter_path, start_epi_itr)
        else:
            with open(output + ".log", 'w') as f:
                pass

        agent_perm, topic_perm = self.random_perm()

        self.warmup(agent_perm, topic_perm)

        # 学習ループ
        for epi_iter in range(start_epi_itr, self.max_epi_itr):
            #  環境のリセット
            self.env.reset()

            #  1エピソード中の reward の保持
            reward_history = []        

            #  各エピソードにおける時間の推移
            for time in range(0, self.env.simulation_time, self.env.time_step):
                print(f"epi_iter, time = {epi_iter}, {time}")

                step = int(time / self.env.time_step)

                #  行動と確率分布の取得
                values, actions, action_log_probs = self.collect(self.batch, step)
                print(f"collect is OK")

                # 報酬の受け取り
                reward = self.env.step(actions, time)
                reward_history.append(reward)
                reward = -reward

                print(f"step is OK")

                #  状態の観測
                agent_perm, topic_perm = self.random_perm()

                obs = self.env.get_observation_mat(agent_perm, topic_perm, obs_size=27)

                print(f"get observation is OK")

                self.insert(self.batch, obs, reward, values, actions, action_log_probs)
                print(f"inser buffer is OK")

            if self.batch == self.trainer.ppo_epoch-1:
                # 学習
                agent.train(obs, obs_topic, actions, pi, reward, next_obs, next_obs_topic)
            
            self.batch = (self.batch + 1) % self.trainer.ppo_epoch

            if epi_iter % 1 == 0:
                #  ログの出力
                #print(f"total_reward = {sum(reward_history)}")
                #print(f"train is {(epi_iter/max_epi_itr)*100}% complited.")
                with open(output + ".log", 'a') as f:
                    f.write(f"{(epi_iter/self.max_epi_itr)*100}%, {-sum(reward_history)}\n")

            #  重みパラメータのバックアップ
            if epi_iter % self.backup_iter == 0:
                agent.save_model(result_dir + 'model_parameter/', actor_weight, critic_weight, V_net_weight, epi_iter)

        #  最適解の取り出し
        df_index = pd.read_csv(learning_data_index_path, index_col=0)
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
        agent.save_model(result_dir + 'model_parameter/', actor_weight, critic_weight, V_net_weight, epi_iter+1)
