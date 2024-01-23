from env import Env
from MAT.env_batch import Env_Batch
from MAT.transformer_policy import TransformerPolicy
from MAT.mat_trainer import MATTrainer
from MAT.shared_buffer import SharedReplayBuffer
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import glob
import random
import copy
from natsort import natsorted
import time as time_module
import datetime
import torch
import optuna
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
    def __init__(self, batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd1, n_embd2, reward_scaling, num_mini_batch, device, max_agent, max_topic):

        # ハイパーパラメーター
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.obs_size = obs_size
        self.n_block = n_block
        self.n_embd1 = n_embd1
        self.n_embd2 = n_embd2
        self.reward_scaling = reward_scaling
        self.num_mini_batch = num_mini_batch

        # 各種パラメーター
        self.device = device
        self.max_agent = max_agent
        self.max_topic = max_topic
        self.N_action = 9
        self.edge_obs_size = 3 ** 2
        self.topic_obs_size = 3
        self.obs_distri_dim = self.obs_size*self.obs_size
        self.obs_dim = self.obs_distri_dim * 4 + self.edge_obs_size * 3 + self.topic_obs_size
        self.obs_info_dim = self.obs_dim - self.obs_distri_dim*4 - self.edge_obs_size*3

        self.random_flag = True
    

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
    

    def get_perm_batch(self, batch_size, random_flag=True):
        
        agent_list = [range(self.max_agent) for _ in range(batch_size)]
        topic_list = [range(self.max_topic) for _ in range(batch_size)]

        if random_flag:
            agent_perm = [random.sample(agent_list[idx], self.max_agent) for idx in range(batch_size)]
            topic_perm = [random.sample(topic_list[idx], self.max_topic) for idx in range(batch_size)]
        else:
            agent_perm = [list(agent_list[idx]) for idx in range(batch_size)]
            topic_perm = [list(topic_list[idx]) for idx in range(batch_size)]

        return np.array(agent_perm, dtype=np.int64), np.array(topic_perm, dtype=np.int64)

    
    #  初期準備
    def warmup(self, buffer, env, batch):
        env.reset()

        agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
        obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)

        buffer.obs_posi[0][batch] = obs_posi
        buffer.obs_publisher[0][batch] = obs_publisher
        buffer.obs_subscriber[0][batch] = obs_subscriber
        buffer.obs_distribution[0][batch] = obs_distribution
        buffer.obs_storage[0][batch] = obs_storage
        buffer.obs_cpu_cycle[0][batch] = obs_cpu_cycle
        buffer.obs_remain_cycle[0][batch] = obs_remain_cycle
        buffer.obs_topic_info[0][batch] = obs_topic_info
        buffer.mask[0][batch] = np.bool_(mask.reshape(self.max_agent*self.max_topic))

        buffer.agent_perm[0][batch] = agent_perm
        buffer.topic_perm[0][batch] = topic_perm

    
    def warmup_batch(self, buffer, env_batch):
        env_batch.reset()

        agent_perm_batch, topic_perm_batch = self.get_perm_batch(env_batch.batch_size, random_flag=self.random_flag)
        obs_posi_batch, obs_publisher_batch, obs_subscriber_batch, obs_distribution_batch, obs_storage_batch, obs_cpu_cycle_batch, obs_remain_cycle_batch, obs_topic_info_batch, mask_batch = env_batch.get_observation_mat(agent_perm_batch, topic_perm_batch, self.obs_size)

        buffer.obs_posi[0] = obs_posi_batch
        buffer.obs_publisher[0] = obs_publisher_batch
        buffer.obs_subscriber[0] = obs_subscriber_batch
        buffer.obs_distribution[0] = obs_distribution_batch
        buffer.obs_storage[0] = obs_storage_batch
        buffer.obs_cpu_cycle[0] = obs_cpu_cycle_batch
        buffer.obs_remain_cycle[0] = obs_remain_cycle_batch
        buffer.obs_topic_info[0] = obs_topic_info_batch
        buffer.mask[0] = np.bool_(mask_batch.reshape(env_batch.batch_size, self.max_agent*self.max_topic))

        buffer.agent_perm[0] = agent_perm_batch
        buffer.topic_perm[0] = topic_perm_batch

    @torch.no_grad()
    def collect(self, trainer, buffer, step, deterministic=False):
        #  TransformerPolicy を学習用に設定
        trainer.prep_rollout()

        obs_posi = buffer.obs_posi[step]
        batch_size = buffer.obs_posi.shape[1]
        obs_client = np.zeros((batch_size, self.max_topic, self.obs_distri_dim*3), dtype=np.float32)
        obs_client[:, :, :self.obs_distri_dim] = buffer.obs_publisher[step]
        obs_client[:, :, self.obs_distri_dim:self.obs_distri_dim*2] = buffer.obs_subscriber[step]
        obs_client[:, :, self.obs_distri_dim*2:self.obs_distri_dim*3] = buffer.obs_distribution[step][:, np.newaxis]
        
        obs_edge = np.zeros((batch_size, self.max_topic, self.edge_obs_size*3), dtype=np.float32)
        obs_edge[:, :, 0:self.edge_obs_size] = buffer.obs_storage[step][:, np.newaxis]
        obs_edge[:, :, self.edge_obs_size:self.edge_obs_size*2] = buffer.obs_cpu_cycle[step][:, np.newaxis]
        obs_edge[:, :, self.edge_obs_size*2:self.edge_obs_size*3] = buffer.obs_remain_cycle[step][:, np.newaxis]
            
        obs_topic_info = buffer.obs_topic_info[step]

        value, action, action_log_prob = trainer.policy.get_actions(obs_posi, obs_client, obs_edge, obs_topic_info, buffer.mask[step], deterministic=deterministic)

        # _t2n: tensor → numpy
        values = np.array(_t2n(value))
        actions = np.array(_t2n(action))
        action_log_probs = np.array(_t2n(action_log_prob))

        return values, actions, action_log_probs
    

    def insert_batch(self, buffer, obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask, rewards, values, actions, action_log_probs, agent_perm, topic_perm):       
        buffer.insert_batch(obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask, actions, action_log_probs, values, rewards, agent_perm, topic_perm)
    

    @torch.no_grad()
    def compute(self, trainer, buffer):
        """Calculate returns for the collected data."""
        #  transformer を評価用にセット
        trainer.prep_rollout()

        obs_posi = buffer.obs_posi[-1]
        obs_client = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim*3), dtype=np.float32)
        obs_client[:, :, :self.obs_distri_dim] = buffer.obs_publisher[-1]
        obs_client[:, :, self.obs_distri_dim:self.obs_distri_dim*2] = buffer.obs_subscriber[-1]
        obs_client[:, :, self.obs_distri_dim*2:self.obs_distri_dim*3] = buffer.obs_distribution[-1][:, np.newaxis]
        
        obs_edge = np.zeros((self.batch_size, self.max_topic, self.edge_obs_size*3), dtype=np.float32)
        obs_edge[:, :, 0:self.edge_obs_size] = buffer.obs_storage[-1][:, np.newaxis]
        obs_edge[:, :, self.edge_obs_size:self.edge_obs_size*2] = buffer.obs_cpu_cycle[-1][:, np.newaxis]
        obs_edge[:, :, self.edge_obs_size*2:self.edge_obs_size*3] = buffer.obs_remain_cycle[-1][:, np.newaxis]
            
        obs_topic_info = buffer.obs_topic_info[-1]

        next_values = trainer.policy.get_values(obs_posi, obs_client, obs_edge, obs_topic_info, buffer.mask[-1])
            
        #  _t2n: tensor から numpy への変換
        next_values = np.array(_t2n(next_values))
        
        buffer.compute_returns_batch(next_values, trainer.value_normalizer)


    def train(self, trainer, buffer):
        #  Transformer を train に設定
        trainer.prep_training()

        trainer.train(buffer)


    def episode_loop(self, simulation_time, time_step, trainer, buffer, batch_size, env_list, reward_history, deternimistic=False):

        #  各エピソードにおける時間の推移
        for time in range(0, simulation_time, time_step):
            step = int(time / time_step)

            #  行動と確率分布の取得
            values_batch, actions_batch, action_log_probs_batch = self.collect(trainer, buffer, step, deterministic=deternimistic)

            reward_batch = np.zeros((batch_size), dtype=np.float32)

            agent_perm_batch = np.zeros((batch_size, self.max_agent), dtype=np.int64)
            topic_perm_batch = np.zeros((batch_size, self.max_topic), dtype=np.int64)

            obs_posi_batch = np.zeros((batch_size, self.max_agent, self.obs_distri_dim), dtype=np.float32)
            obs_publisher_batch = np.zeros((batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_subscriber_batch = np.zeros((batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_distribution_batch = np.zeros((batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_storage_batch = np.zeros((batch_size, self.edge_obs_size), dtype=np.float32)
            obs_cpu_cycle_batch = np.zeros((batch_size, self.edge_obs_size), dtype=np.float32)
            obs_remain_cycle_batch = np.zeros((batch_size, self.edge_obs_size), dtype=np.float32)
            obs_topic_info_batch = np.zeros((batch_size, self.max_topic, self.topic_obs_size), dtype=np.float32)
            mask_batch = np.zeros((batch_size, self.max_agent, self.max_topic), dtype=np.bool)

            # 報酬の受け取り
            for idx in range(batch_size):
                env = env_list[idx]
                reward = env.step(actions_batch[idx][buffer.mask[step][idx]], buffer.agent_perm[step][idx], buffer.topic_perm[step][idx], time)
                reward_history[idx].append(reward)
                if self.reward_scaling == True:
                    reward_batch[idx] = (-reward / 200) + 1
                else:
                    reward_batch[idx] = (-reward)

                #  状態の観測
                #  ランダムな順にいつか改修
                agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
                agent_perm_batch[idx] = agent_perm
                topic_perm_batch[idx] = topic_perm

                obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
                obs_posi_batch[idx] = obs_posi
                obs_publisher_batch[idx] = obs_publisher
                obs_subscriber_batch[idx] = obs_subscriber
                obs_distribution_batch[idx] = obs_distribution
                obs_storage_batch[idx] = obs_storage
                obs_cpu_cycle_batch[idx] = obs_cpu_cycle
                obs_remain_cycle_batch[idx] = obs_remain_cycle
                obs_topic_info_batch[idx] = obs_topic_info
                mask_batch[idx] = mask
           
            self.insert_batch(buffer, obs_posi_batch, obs_publisher_batch, obs_subscriber_batch, obs_distribution_batch, obs_storage_batch, obs_cpu_cycle_batch, obs_remain_cycle_batch, obs_topic_info_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)
    

    def episode_loop_batch(self, simulation_time, time_step, trainer, buffer, batch_size, env_batch, env_list, reward_history, deternimistic=False):

        #  各エピソードにおける時間の推移
        for time in range(0, simulation_time, time_step):
            step = int(time / time_step)

            #  行動と確率分布の取得
            values_batch, actions_batch, action_log_probs_batch = self.collect(trainer, buffer, step, deterministic=deternimistic)

            old_reward_batch = np.zeros((batch_size), dtype=np.float32)

            # 報酬の受け取り
            for idx in range(batch_size):
                env = env_list[idx]
                old_reward = env.step(actions_batch[idx][buffer.mask[step][idx]], buffer.agent_perm[step][idx], buffer.topic_perm[step][idx], time)

                if self.reward_scaling == True:
                    old_reward_batch[idx] = (-old_reward / 200) + 1
                else:
                    old_reward_batch[idx] = (-old_reward)

            reward_batch = env_batch.step(actions_batch[buffer.mask[step]], buffer.agent_perm[step], buffer.topic_perm[step], time)
            reward_batch = reward_batch * (-1)

            flag = True
            for batch_idx in range(self.batch_size):
                if old_reward_batch[batch_idx] != reward_batch[batch_idx]:
                    # print(f"opt = {reward_batch[batch_idx]}, old = {old_reward_batch[batch_idx]}")
                    # print(f"error")
                    flag = False

            if flag:
                print(f"OK")

            # self.check_batch_env(env_batch, env_list, time, time_step)

            reward_history.append(reward_batch)
            reward_batch = reward_batch * -1

            agent_perm_batch, topic_perm_batch = self.get_perm_batch(batch_size, random_flag=self.random_flag)
            obs_posi_batch, obs_publisher_batch, obs_subscriber_batch, obs_distribution_batch, obs_storage_batch, obs_cpu_cycle_batch, obs_remain_cycle_batch, obs_topic_info_batch, mask_batch = env_batch.get_observation_mat(agent_perm_batch, topic_perm_batch, self.obs_size)
           
            self.insert_batch(buffer, obs_posi_batch, obs_publisher_batch, obs_subscriber_batch, obs_distribution_batch, obs_storage_batch, obs_cpu_cycle_batch, obs_remain_cycle_batch, obs_topic_info_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)
        

    def cal_nearest_server_reward(self, index_path):
        nearest_reward = 0

        env = Env(index_path)
        simulation_time = env.simulation_time
        time_step = env.time_step

        agent_perm, topic_perm = self.get_perm(random_flag=False)

        for time in range(0, simulation_time, time_step):
            obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, obs_size=self.obs_size)
            mask = np.bool_(mask.reshape(-1))
            actions = env.get_near_action(agent_perm, topic_perm)

            nearest_reward += env.step(actions[mask], agent_perm, topic_perm, time)

        return nearest_reward

    
    def train_single_env(self, start_epi_itr, max_epi_itr, learning_data_index_path, result_dir, output, transformer_weight, backup_itr, load_parameter_path=None):
        if not os.path.isdir(result_dir + "model_parameter"):
            sys.exit("結果を格納するディレクトリ" + result_dir + "model_parameter が作成されていません。")

        # 環境のインスタンスの生成
        env = Env(learning_data_index_path)
        simulation_time = env.simulation_time
        time_step = env.time_step

        if simulation_time % time_step == 0:
            episode_length = int(simulation_time / time_step)
        else:
            sys.exit("simulation_time が time_step の整数倍になっていません")

        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.edge_obs_size, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd1, self.n_embd2, device=self.device, multi=False)

        trainer = MATTrainer(policy, self.ppo_epoch, self.num_mini_batch, self.device)

        buffer = SharedReplayBuffer(episode_length, self.batch_size, self.max_agent, self.max_topic, self.obs_dim, self.N_action)

        if load_parameter_path is not None:
            policy.restore(load_parameter_path)
            if start_epi_itr == 0:
                with open(output + ".log", 'w') as f:
                    pass
        else:
            with open(output + ".log", 'w') as f:
                pass

        timezone_jst = datetime.timezone(datetime.timedelta(hours=9))
        start_process = datetime.datetime.now(timezone_jst)
        print(f"開始時刻: {start_process}")

        env_list = [Env(learning_data_index_path) for _ in range(self.batch_size)]

        # 学習ループ
        for epi_iter in range(start_epi_itr, max_epi_itr):
            start_time = time_module.perf_counter()

            #  1エピソード中の reward の保持
            reward_history = [[] for _ in range(self.batch_size)]

            #  環境のリセット            
            for idx in range(self.batch_size):
                env = env_list[idx]

                self.warmup(buffer, env, idx)

            self.episode_loop(simulation_time, time_step, trainer, buffer, self.batch_size, env_list, reward_history) 

            self.compute(trainer, buffer)

            self.train(trainer, buffer)
            
            if epi_iter % 1 == 0:
                #  ログの出力
                with open(output + ".log", 'a') as f:
                    reward_average = 0
                    for idx in range(self.batch_size):
                        reward_average += sum(reward_history[idx])/self.batch_size

                    f.write(f"{(epi_iter/max_epi_itr)*100}%, {reward_average * -1}\n")

            #  重みパラメータのバックアップ
            if epi_iter % backup_itr == 0:
                policy.save(result_dir + 'model_parameter', transformer_weight, epi_iter)

            end_time = time_module.perf_counter()

            if epi_iter == 0:
                print(f"1 step time = {end_time - start_time}")

                process_time = datetime.timedelta(seconds=(end_time - start_time)*max_epi_itr)
                finish_time = start_process + process_time
                print(f"終了予定時刻: {finish_time}")

        #  最適解の取り出し
        df_index = pd.read_csv(learning_data_index_path, index_col=0)
        opt = df_index.at['data', 'opt']

        #  近傍サーバを選択した際の遅延を計算
        nearest_reward = self.cal_nearest_server_reward(learning_data_index_path)

        train_curve = read_train_curve(output + ".log")

        #  学習曲線の描画
        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        wind.grid()
        wind.plot(train_curve, linewidth=1, label='')
        wind.axhline(y=-opt, c='r', label='opt')
        wind.axhline(y=-nearest_reward, c='g', label='nearest')
        fig.savefig(output + ".png")

        #  重みパラメータの保存
        policy.save(result_dir + 'model_parameter', transformer_weight, epi_iter+1)


    def train_multi_env(self, sample_data, start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, result_dir, output, transformer_weight, backup_itr, load_parameter_path=None):
        test_iter = 10

        train_dir_path = os.path.join(learning_data_index_dir, "*")
        train_index_path = natsorted(glob.glob(train_dir_path))

        test_dir_path = os.path.join(test_data_index_dir, "*")
        test_index_path = natsorted(glob.glob(test_dir_path))
        
        test_env_list = []
        for idx in range(len(test_index_path)):
            test_env_list.append(Env(test_index_path[idx]))

        tmp_env = test_env_list[0]
        simulation_time = tmp_env.simulation_time
        time_step = tmp_env.time_step

        episode_length = int(simulation_time / time_step)
            
        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.edge_obs_size, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd1, self.n_embd2, device=self.device, multi=False)
        trainer = MATTrainer(policy, self.ppo_epoch, self.num_mini_batch, self.device)

        buffer = SharedReplayBuffer(episode_length, self.batch_size, self.max_agent, self.max_topic, self.obs_dim, self.N_action)
        test_buffer = SharedReplayBuffer(episode_length, len(test_env_list), self.max_agent, self.max_topic, self.obs_dim, self.N_action)

        timezone_jst = datetime.timezone(datetime.timedelta(hours=9))
        start_process = datetime.datetime.now(timezone_jst)
        print(f"開始時刻: {start_process}")

        if load_parameter_path is not None:
            policy.restore(load_parameter_path)
            if start_epi_itr == 0:
                with open(output + ".log", 'w') as f:
                    pass
                
                for idx in range(len(test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'w') as f:
                        pass
        else:
            with open(output + ".log", 'w') as f:
                pass

            for idx in range(len(test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'w') as f:
                        pass

        # 学習ループ
        for epi_iter in range(start_epi_itr, max_epi_itr):
            start_time = time_module.perf_counter()

            #  環境のリセット
            env_list_shuffle = []
            train_index_path_shuffle = random.sample(train_index_path, sample_data)
            for idx in range(sample_data):
                for _ in range(int(self.batch_size / sample_data)):
                    env_list_shuffle.append(Env(train_index_path_shuffle[idx]))
                    
            #  1エピソード中の reward の保持
            reward_history = [[] for _ in range(self.batch_size)]

            warmup_start = time_module.perf_counter()

            #  環境のリセット            
            for idx in range(self.batch_size):
                env = env_list_shuffle[idx]

                self.warmup(buffer, env, idx)

            warmup_end = time_module.perf_counter()

            self.episode_loop(simulation_time, time_step, trainer, buffer, self.batch_size, env_list_shuffle, reward_history)
                        
            compute_start = time_module.perf_counter()

            self.compute(trainer, buffer)

            compute_end = time_module.perf_counter()

            train_start = time_module.perf_counter()

            self.train(trainer, buffer)

            train_end = time_module.perf_counter()
            
            if epi_iter % 1 == 0:
                #  ログの出力
                with open(output + ".log", 'a') as f:
                    reward_average = 0
                    for idx in range(self.batch_size):
                        reward_average += sum(reward_history[idx])/self.batch_size

                    f.write(f"{(epi_iter/max_epi_itr)*100}%, {reward_average * -1}\n")

            if epi_iter % test_iter == 0 or (epi_iter+1) == max_epi_itr:
                test_start = time_module.perf_counter()

                for idx in range(len(test_env_list)):
                    test_env = test_env_list[idx]

                    self.warmup(test_buffer, test_env, idx)

                #  各エピソードにおける時間の推移
                reward_history_test = [[] for _ in range(len(test_env_list))]

                self.episode_loop(simulation_time, time_step, trainer, test_buffer, len(test_env_list), test_env_list, reward_history_test, deternimistic=False)

                for idx in range(len(test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'a') as f:
                        f.write(f"{(epi_iter/max_epi_itr)*100}%, {-sum(reward_history_test[idx])}\n")

                test_end = time_module.perf_counter()

            #  重みパラメータのバックアップ
            if epi_iter % backup_itr == 0:
                policy.save(result_dir + 'model_parameter', transformer_weight, epi_iter)

            end_time = time_module.perf_counter()

            if epi_iter == 0:
                #print(f"1 step time = {end_time - start_time}")
                print(f"1 step time = {train_end - start_time}")
                print(f"shuffle env time = {warmup_start - start_time}")
                print(f"warmup time = {warmup_end - warmup_start}")
                print(f"episode loop time = {compute_start - warmup_end}")
                print(f"compute time = {compute_end - compute_start}")
                print(f"train time = {train_end - train_start}")

                process_time = datetime.timedelta(seconds=(end_time - start_time - (test_end -test_start))*max_epi_itr + (test_end - test_start)*(max_epi_itr / test_iter))
                finish_time = start_process + process_time
                print(f"終了予定時刻: {finish_time}")

        #  重みパラメータの保存
        policy.save(result_dir + 'model_parameter', transformer_weight, epi_iter+1)


    def debug_multi_env(self, sample_data, start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, result_dir, output, transformer_weight, backup_itr, load_parameter_path=None):
        test_iter = 10

        train_dir_path = os.path.join(learning_data_index_dir, "*")
        train_index_path = natsorted(glob.glob(train_dir_path))

        test_dir_path = os.path.join(test_data_index_dir, "*")
        test_index_path = natsorted(glob.glob(test_dir_path))

        test_env_batch = Env_Batch(test_index_path)

        tmp_env = Env(train_index_path[0])

        simulation_time = tmp_env.simulation_time
        time_step = tmp_env.time_step

        episode_length = int(simulation_time / time_step)
        
        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.edge_obs_size, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd1, self.n_embd2, device=self.device, multi=False)
        trainer = MATTrainer(policy, self.ppo_epoch, self.num_mini_batch, self.device)

        buffer = SharedReplayBuffer(episode_length, self.batch_size, self.max_agent, self.max_topic, self.obs_dim, self.N_action)
        test_buffer = SharedReplayBuffer(episode_length, test_env_batch.batch_size, self.max_agent, self.max_topic, self.obs_dim, self.N_action)

        timezone_jst = datetime.timezone(datetime.timedelta(hours=9))
        start_process = datetime.datetime.now(timezone_jst)
        print(f"開始時刻: {start_process}")

        if load_parameter_path is not None:
            policy.restore(load_parameter_path)
            if start_epi_itr == 0:
                with open(output + ".log", 'w') as f:
                    pass
                
                for idx in range(test_env_batch.batch_size):
                    with open(output + "_test" + str(idx) + ".log", 'w') as f:
                        pass
        else:
            with open(output + ".log", 'w') as f:
                pass

            for idx in range(test_env_batch.batch_size):
                    with open(output + "_test" + str(idx) + ".log", 'w') as f:
                        pass

        # 学習ループ
        for epi_iter in range(start_epi_itr, max_epi_itr):
            start_time = time_module.perf_counter()

            #  環境のリセット
            env_list_shuffle = []
            train_index_path_shuffle = random.sample(train_index_path, sample_data)

            for idx in range(sample_data):
                for _ in range(int(self.batch_size / sample_data)):
                    env_list_shuffle.append(Env(train_index_path_shuffle[idx]))

            env_batch = Env_Batch(train_index_path_shuffle)
                    
            #  1エピソード中の reward の保持
            reward_history = [[] for _ in range(self.batch_size)]

            warmup_start = time_module.perf_counter()

            self.warmup_batch(buffer, env_batch)

            warmup_end = time_module.perf_counter()

            self.episode_loop_batch(simulation_time, time_step, trainer, buffer, self.batch_size, env_batch, env_list_shuffle, reward_history)
                        
            compute_start = time_module.perf_counter()

            self.compute(trainer, buffer)

            compute_end = time_module.perf_counter()

            train_start = time_module.perf_counter()

            self.train(trainer, buffer)

            train_end = time_module.perf_counter()
            
            if epi_iter % 1 == 0:
                #  ログの出力
                with open(output + ".log", 'a') as f:
                    reward_average = 0
                    for idx in range(self.batch_size):
                        reward_average += sum(reward_history[idx])/self.batch_size

                    f.write(f"{(epi_iter/max_epi_itr)*100}%, {reward_average * -1}\n")

            if epi_iter % test_iter == 0 or (epi_iter+1) == max_epi_itr:
                test_start = time_module.perf_counter()

                self.warmup_batch(test_buffer, test_env_batch)

                #  各エピソードにおける時間の推移
                reward_history_test = [[] for _ in range(test_env_batch.batch_size)]

                self.episode_loop_batch(simulation_time, time_step, trainer, test_buffer, test_env_batch.batch_size, test_env_batch, [], reward_history_test, deternimistic=False)

                for idx in range(test_env_batch.batch_size):
                    with open(output + "_test" + str(idx) + ".log", 'a') as f:
                        f.write(f"{(epi_iter/max_epi_itr)*100}%, {-sum(reward_history_test[idx])}\n")

                test_end = time_module.perf_counter()

            #  重みパラメータのバックアップ
            if epi_iter % backup_itr == 0:
                policy.save(result_dir + 'model_parameter', transformer_weight, epi_iter)

            end_time = time_module.perf_counter()

            if epi_iter == 0:
                #print(f"1 step time = {end_time - start_time}")
                print(f"1 step time = {train_end - start_time}")
                print(f"shuffle env time = {warmup_start - start_time}")
                print(f"warmup time = {warmup_end - warmup_start}")
                print(f"episode loop time = {compute_start - warmup_end}")
                print(f"compute time = {compute_end - compute_start}")
                print(f"train time = {train_end - train_start}")

                process_time = datetime.timedelta(seconds=(end_time - start_time - (test_end -test_start))*max_epi_itr + (test_end - test_start)*(max_epi_itr / test_iter))
                finish_time = start_process + process_time
                print(f"終了予定時刻: {finish_time}")

        #  重みパラメータの保存
        policy.save(result_dir + 'model_parameter', transformer_weight, epi_iter+1)


    def tuning_single_env(self, trial, start_epi_itr, max_epi_itr, index_path, log_name_base, process_name):
        with open(log_name_base + "trail" + str(trial.number) + "_learning_log_" + process_name + ".log", "w") as f:
            pass

        # 環境のインスタンスの生成
        env = Env(index_path)
        simulation_time = env.simulation_time
        time_step = env.time_step

        if simulation_time % time_step == 0:
            episode_length = int(simulation_time / time_step)
        else:
            sys.exit("simulation_time が time_step の整数倍になっていません")

        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.edge_obs_size, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd1, self.n_embd2, device=self.device, multi=False)

        trainer = MATTrainer(policy, self.ppo_epoch, self.num_mini_batch, self.device)

        buffer = SharedReplayBuffer(episode_length, self.batch_size, self.max_agent, self.max_topic, self.obs_dim, self.N_action)

        env_list = [Env(index_path) for _ in range(self.batch_size)]

        # 学習ループ
        for epi_iter in range(start_epi_itr, max_epi_itr):
            #  1エピソード中の reward の保持
            reward_history = [[] for _ in range(self.batch_size)]

            #  環境のリセット            
            for idx in range(self.batch_size):
                env = env_list[idx]

                self.warmup(buffer, env, idx)

            self.episode_loop(simulation_time, time_step, trainer, buffer, self.batch_size, env_list, reward_history) 

            self.compute(trainer, buffer)

            self.train(trainer, buffer)

            reward_average = 0
            for idx in range(self.batch_size):
                reward_average += sum(reward_history[idx]) / self.batch_size
            
            with open(log_name_base + "trail" + str(trial.number) + "_learning_log_" + process_name + ".log", "a") as f:
                f.write(f"{epi_iter}/{max_epi_itr}, {reward_average}\n")

            trial.report(reward_average, epi_iter)

            if trial.should_prune():
                raise optuna.TrialPruned()

        reward_average = 0
        for idx in range(self.batch_size):
            reward_average += sum(reward_history[idx]) / self.batch_size

        return reward_average
    

    def tuning_multi_env(self, trial, start_epi_itr, max_epi_itr, index_dir, test_dir, log_dir, log_name_base, process_name):
        test_iter = 10

        train_dir_path = os.path.join(index_dir, "*")
        train_index_path = natsorted(glob.glob(train_dir_path))

        test_dir_path = os.path.join(test_dir, "*")
        test_index_path = natsorted(glob.glob(test_dir_path))

        test_env_list = []
        for idx in range(len(test_index_path)):
            test_env_list.append(Env(test_index_path[idx]))

        simulation_time = test_env_list[0].simulation_time
        time_step = test_env_list[0].time_step

        episode_length = int(simulation_time / time_step)

        with open(log_dir + "trial" + str(trial.number) + "/" + log_name_base + "trail" + str(trial.number) + "_learning_log_" + process_name + ".log", "w") as f:
            pass

        for idx in range(len(test_env_list)):
            with open(log_dir + "trial" + str(trial.number) + "/" + log_name_base + "trail" + str(trial.number) + "_test" + str(idx) + "_" + process_name + ".log", "w") as f:
                pass

        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.edge_obs_size, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd1, self.n_embd2, device=self.device, multi=False)
        trainer = MATTrainer(policy, self.ppo_epoch, self.num_mini_batch, self.device)

        buffer = SharedReplayBuffer(episode_length, self.batch_size, self.max_agent, self.max_topic, self.obs_dim, self.N_action)
        test_buffer = SharedReplayBuffer(episode_length, len(test_env_list), self.max_agent, self.max_topic, self.obs_dim, self.N_action)

        # 学習ループ
        for epi_iter in range(start_epi_itr, max_epi_itr):
            #  環境のリセット
            env_list_shuffle = []
            train_index_path_shuffle = random.sample(train_index_path, self.batch_size)
            for idx in range(self.batch_size):
                env_list_shuffle.append(Env(train_index_path_shuffle[idx]))

            #  1エピソード中の reward の保持
            reward_history = [[] for _ in range(self.batch_size)]

            #  環境のリセット            
            for idx in range(self.batch_size):
                env = env_list_shuffle[idx]

                self.warmup(buffer, env, idx)

            self.episode_loop(simulation_time, time_step, trainer, buffer, self.batch_size, env_list_shuffle, reward_history) 

            self.compute(trainer, buffer)

            self.train(trainer, buffer)

            reward_average = 0
            for idx in range(self.batch_size):
                reward_average += sum(reward_history[idx]) / self.batch_size
            
            with open(log_dir + "trial" + str(trial.number) + "/" + log_name_base + "trail" + str(trial.number) + "_learning_log_" + process_name + ".log", "a") as f:
                f.write(f"{epi_iter}/{max_epi_itr}, {reward_average}\n")

            if epi_iter % test_iter == 0 or (epi_iter+1) == max_epi_itr:
                for idx in range(len(test_env_list)):
                    test_env = test_env_list[idx]

                    self.warmup(test_buffer, test_env, idx)
                
                reward_history_test = [[] for _ in range(len(test_env_list))]

                self.episode_loop(simulation_time, time_step, trainer, test_buffer, len(test_env_list), test_env_list, reward_history_test, deternimistic=False)

                reward_test_average = 0
                for idx in range(len(test_env_list)):
                    reward_test_average += sum(reward_history_test[idx]) / len(test_env_list)

                for idx in range(len(test_env_list)):
                    with open(log_dir + "trial" + str(trial.number) + "/" + log_name_base + "trail" + str(trial.number) + "_test" + str(idx) + "_" + process_name + ".log", "a") as f:
                        f.write(f"{epi_iter}/{max_epi_itr}, {sum(reward_history_test[idx])}\n")

                trial.report(reward_test_average, epi_iter)

                if trial.should_prune():
                    raise optuna.TrialPruned()

        reward_test_average = 0
        for idx in range(len(test_env_list)):
            reward_test_average += sum(reward_history_test[idx]) / len(test_env_list)

        return reward_test_average
    

    def execute_single_env(self,  data_index_path, load_parameter_path):
        env_list = []
        for _ in range(self.batch_size):
            env_list.append(Env(data_index_path))

        simulation_time = env_list[0].simulation_time
        time_step = env_list[0].time_step 

        if simulation_time % time_step == 0:
            episode_length = int(simulation_time / time_step)
        else:
            sys.exit("simulation_time が time_step の整数倍になっていません")

        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.edge_obs_size, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd1, self.n_embd2, self.device, multi=False)

        trainer = MATTrainer(policy, self.ppo_epoch, self.num_mini_batch, self.device)

        buffer = SharedReplayBuffer(episode_length, self.batch_size, self.max_agent, self.max_topic, self.obs_dim, self.N_action)

        policy.restore(load_parameter_path)

        for idx in range(self.batch_size):
            env = env_list[idx]
            self.warmup(buffer, env, idx)

        reward_history = [[] for _ in range(self.batch_size)]

        self.episode_loop(simulation_time, time_step, trainer, buffer, self.batch_size, env_list, reward_history, deternimistic=False)

        return reward_history
    

    def check_batch_env(self, env_batch, env_list, time, time_step):
        
        # client_x のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for client_idx in range(env_list[batch_idx].num_client):
                if env_batch.client_x[batch_idx][client_idx] != env_list[batch_idx].clients[client_idx].x:
                    Flag = False
            
        if Flag:
            print(f"client_x is OK!")
        else:
            print(f"client_x is error")

        # client_y のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for client_idx in range(env_list[batch_idx].num_client):
                if env_batch.client_y[batch_idx][client_idx] != env_list[batch_idx].clients[client_idx].y:
                    Flag = False
            
        if Flag:
            print(f"client_y is OK!")
        else:
            print(f"client_y is error")

        # client_pub_topic のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for client_idx in range(env_list[batch_idx].num_client):
                for topic_idx in range(env_list[batch_idx].num_topic):
                    if env_batch.client_pub_topic[batch_idx][client_idx][topic_idx] != env_list[batch_idx].clients[client_idx].pub_topic[topic_idx]:
                        Flag = False
            
        if Flag:
            print(f"client_pub_topic is OK!")
        else:
            print(f"client_pub_topic is error")

        # client_sub_topic のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for client_idx in range(env_list[batch_idx].num_client):
                for topic_idx in range(env_list[batch_idx].num_topic):
                    if env_batch.client_sub_topic[batch_idx][client_idx][topic_idx] != env_list[batch_idx].clients[client_idx].sub_topic[topic_idx]:
                        Flag = False
            
        if Flag:
            print(f"client_sub_topic is OK!")
        else:
            print(f"client_sub_topic is error")

        # client_pub_edge のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for client_idx in range(env_list[batch_idx].num_client):
                for topic_idx in range(env_list[batch_idx].num_topic):
                    for edge_idx in range(env_list[batch_idx].num_edge):
                        if env_batch.client_pub_edge[batch_idx][client_idx][topic_idx][edge_idx] == 1:
                            if env_list[batch_idx].clients[client_idx].pub_edge[topic_idx] != edge_idx:
                                Flag = False
                        
                        else:
                            if env_list[batch_idx].clients[client_idx].pub_edge[topic_idx] == edge_idx:
                                Flag = False
            
        if Flag:
            print(f"client_pub_edge is OK!")
        else:
            print(f"client_pub_edge is error")

        # client_sub_edge のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for client_idx in range(env_list[batch_idx].num_client):
                for topic_idx in range(env_list[batch_idx].num_topic):
                    for edge_idx in range(env_list[batch_idx].num_edge):
                        if env_batch.client_sub_edge[batch_idx][client_idx][edge_idx] == 1:
                            if env_list[batch_idx].clients[client_idx].sub_edge[topic_idx] != edge_idx:
                                Flag = False

                        else:
                            if env_list[batch_idx].clients[client_idx].sub_edge[topic_idx] == edge_idx:
                                Flag = False
            
        if Flag:
            print(f"client_sub_edge is OK!")
        else:
            print(f"client_sub_edge is error")
                
        # edge_x のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                if env_batch.edge_x[batch_idx][edge_idx] != env_list[batch_idx].all_edge[edge_idx].x:
                    Flag = False
            
        if Flag:
            print(f"edge_x is OK!")
        else:
            print(f"edge_x is error")

        # edge_y のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                if env_batch.edge_y[batch_idx][edge_idx] != env_list[batch_idx].all_edge[edge_idx].y:
                    Flag = False
            
        if Flag:
            print(f"edge_y is OK!")
        else:
            print(f"edge_y is error")

        # edge_max_volume のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                if env_batch.edge_max_volume[batch_idx][edge_idx] != env_list[batch_idx].all_edge[edge_idx].max_volume:
                    Flag = False
            
        if Flag:
            print(f"edge_max_volume is OK!")
        else:
            print(f"edge_max_volume is error")

        # edge_used_volume のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                for topic_idx in range(env_list[batch_idx].num_topic):
                    if env_batch.edge_used_volume[batch_idx][edge_idx][topic_idx] != env_list[batch_idx].all_edge[edge_idx].used_volume[topic_idx]:
                        Flag = False
            
        if Flag:
            print(f"edge_used_volume is OK!")
        else:
            print(f"edge_used_volume is error")

        # edge_deploy_topic のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                for topic_idx in range(env_list[batch_idx].num_topic):
                    if env_batch.edge_deploy_topic[batch_idx][edge_idx][topic_idx] != env_list[batch_idx].all_edge[edge_idx].deploy_topic[topic_idx]:
                        Flag = False
            
        if Flag:
            print(f"edge_deploy_topic is OK!")
        else:
            print(f"edge_deploy_topic is error")

        # edge_cpu_cycle のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                if env_batch.edge_cpu_cycle[batch_idx][edge_idx] != env_list[batch_idx].all_edge[edge_idx].cpu_cycle:
                    Flag = False
            
        if Flag:
            print(f"edge_cpu_cycle is OK!")
        else:
            print(f"edge_cpu_cycle is error")

        # edge_power_allocation のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                if env_batch.edge_power_allocation[batch_idx][edge_idx] != env_list[batch_idx].all_edge[edge_idx].power_allocation:
                    Flag = False
            
        if Flag:
            print(f"edge_power_allocation is OK!")
        else:
            print(f"edge_power_allocation is error")

        # edge_used_publisher のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                for topic_idx in range(env_list[batch_idx].num_topic):
                    if env_batch.edge_used_publisher[batch_idx][edge_idx][topic_idx] != env_list[batch_idx].all_edge[edge_idx].used_publishers[topic_idx]:
                        Flag = False
            
        if Flag:
            print(f"edge_used_publisher is OK!")
        else:
            print(f"edge_used_publisher is error")

        # edge_remain_cycle のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for edge_idx in range(env_list[batch_idx].num_edge):
                if env_batch.edge_remain_cycle[batch_idx][edge_idx] != env_list[batch_idx].all_edge[edge_idx].remain_cycle:
                    Flag = False
            
        if Flag:
            print(f"edge_remain_cycle is OK!")
        else:
            print(f"edge_remain_cycle is error")

        # topic_save_period のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for topic_idx in range(env_list[batch_idx].num_topic):
                if env_batch.topic_save_period[batch_idx][topic_idx] != env_list[batch_idx].all_topic[topic_idx].save_period:
                    Flag = False
            
        if Flag:
            print(f"topic_save_period is OK!")
        else:
            print(f"topic_save_period is error")

        # topic_publish_rate のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for topic_idx in range(env_list[batch_idx].num_topic):
                if env_batch.topic_publish_rate[batch_idx][topic_idx] != env_list[batch_idx].all_topic[topic_idx].publish_rate:
                    Flag = False
            
        if Flag:
            print(f"topic_publish_rate is OK!")
        else:
            print(f"topic_publish_rate is error")

        # topic_data_size のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for topic_idx in range(env_list[batch_idx].num_topic):
                if env_batch.topic_data_size[batch_idx][topic_idx] != env_list[batch_idx].all_topic[topic_idx].data_size:
                    Flag = False
            
        if Flag:
            print(f"topic_data_size is OK!")
        else:
            print(f"topic_data_size is error")

        # topic_require_cycle のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for topic_idx in range(env_list[batch_idx].num_topic):
                if env_batch.topic_require_cycle[batch_idx][topic_idx] != env_list[batch_idx].all_topic[topic_idx].require_cycle:
                    Flag = False
            
        if Flag:
            print(f"topic_require_cycle is OK!")
        else:
            print(f"topic_require_cycyle is error")

        # topic_volume のチェック
        Flag = True
        for batch_idx in range(self.batch_size):
            for topic_idx in range(env_list[batch_idx].num_topic):
                if env_batch.topic_volume[batch_idx][topic_idx] != env_list[batch_idx].all_topic[topic_idx].volume:
                    Flag = False
            
        if Flag:
            print(f"topic_volume is OK!")
        else:
            print(f"topic_volume is error")

        # topic_num_client_history のチェック
        Flag = True
        now_step = int(time/time_step)
        for batch_idx in range(self.batch_size):
            for topic_idx in range(env_list[batch_idx].num_topic):
                save_period = env_list[topic_idx].all_topic[topic_idx].save_period
                start_step = max(now_step - int(save_period/time_step), 0)

                if sum(env_batch.topic_num_client_history[batch_idx][topic_idx][start_step:now_step+1]) != env_list[batch_idx].all_topic[topic_idx].total_num_client:
                    Flag = False

        if Flag:
            print(f"topic_num_clinet_history is OK!")
        else:
            print(f"topic_num_clinet_history is error")

    def check_batch_env_obs(self, env_batch, env_list):
        # 共通で使用
        agent_perm_batch, topic_perm_batch = self.get_perm_batch(random_flag=self.random_flag)

        # env_batch
        opt_obs_posi_batch, opt_obs_publisher_batch, opt_obs_subscriber_batch, opt_obs_distribution_batch, opt_obs_storage_batch, opt_obs_cpu_cycle_batch, opt_obs_remain_cycle_batch, opt_obs_topic_info_batch, opt_mask_batch = env_batch.get_observation_mat(agent_perm_batch, topic_perm_batch, self.obs_size)

        # env_list
        obs_posi_batch = np.zeros((self.batch_size, self.max_agent, self.obs_distri_dim), dtype=np.float32)
        obs_publisher_batch = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
        obs_subscriber_batch = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
        obs_distribution_batch = np.zeros((self.batch_size, self.obs_distri_dim), dtype=np.float32)
        obs_storage_batch = np.zeros((self.batch_size, self.edge_obs_size), dtype=np.float32)
        obs_cpu_cycle_batch = np.zeros((self.batch_size, self.edge_obs_size), dtype=np.float32)
        obs_remain_cycle_batch = np.zeros((self.batch_size, self.edge_obs_size), dtype=np.float32)
        obs_topic_info_batch = np.zeros((self.batch_size, self.max_topic, self.topic_obs_size), dtype=np.float32)
        mask_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic), dtype=np.bool)

        for idx in range(self.batch_size):
            env = env_list[idx]

            obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask = env.get_observation_mat(agent_perm_batch[idx], topic_perm_batch[idx], self.obs_size)
            obs_posi_batch[idx] = obs_posi
            obs_publisher_batch[idx] = obs_publisher
            obs_subscriber_batch[idx] = obs_subscriber
            obs_distribution_batch[idx] = obs_distribution
            obs_storage_batch[idx] = obs_storage
            obs_cpu_cycle_batch[idx] = obs_cpu_cycle
            obs_remain_cycle_batch[idx] = obs_remain_cycle
            obs_topic_info_batch[idx] = obs_topic_info
            mask_batch[idx] = mask

        # obs_posi のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for agent_idx in range(self.max_agent):
                for channel_idx in range(self.obs_distri_dim):
                    if opt_obs_posi_batch[batch_idx][agent_idx][channel_idx] != obs_posi_batch[batch_idx][agent_idx][channel_idx]:
                        flag = False

        if flag:
            print(f"obs_posi_batch is OK !!!!")
        else:
            print(f"obs_posi_batch is error !!!")

        # obs_publisher のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for topic_idx in range(self.max_topic):
                for channel_idx in range(self.obs_distri_dim):
                    if opt_obs_publisher_batch[batch_idx][topic_idx][channel_idx] != obs_publisher_batch[batch_idx][topic_idx][channel_idx]:
                        flag = False

        if flag:
            print(f"obs_publisher_batch is OK !!!!")
        else:
            print(f"obs_publisher_batch is error !!!")

        # obs_subscriber のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for topic_idx in range(self.max_topic):
                for channel_idx in range(self.obs_distri_dim):
                    if opt_obs_subscriber_batch[batch_idx][topic_idx][channel_idx] != obs_subscriber_batch[batch_idx][topic_idx][channel_idx]:
                        flag = False

        if flag:
            print(f"obs_subscriber_batch is OK !!!!")
        else:
            print(f"obs_subscriber_batch is error !!!")

        # obs_distribution のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for channel_idx in range(self.obs_distri_dim):
                if opt_obs_distribution_batch[batch_idx][channel_idx] != obs_distribution_batch[batch_idx][channel_idx]:
                    flag = False

        if flag:
            print(f"obs_distribution_batch is OK !!!!")
        else:
            print(f"obs_distribution_batch is error !!!")

        # obs_storage のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for channel_idx in range(self.edge_obs_size):
                if opt_obs_storage_batch[batch_idx][channel_idx] != obs_storage_batch[batch_idx][channel_idx]:
                    flag = False

        if flag:
            print(f"obs_storage_batch is OK !!!!")
        else:
            print(f"obs_storage_batch is error !!!")

        # obs_cpu_cycle のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for channel_idx in range(self.edge_obs_size):
                if opt_obs_cpu_cycle_batch[batch_idx][channel_idx] != obs_cpu_cycle_batch[batch_idx][channel_idx]:
                    flag = False

        if flag:
            print(f"obs_cpu_cycle_batch is OK !!!!")
        else:
            print(f"obs_cpu_cycle_batch is error !!!")

        # obs_remain_cycle のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for channel_idx in range(self.edge_obs_size):
                if opt_obs_remain_cycle_batch[batch_idx][channel_idx] != obs_remain_cycle_batch[batch_idx][channel_idx]:
                    flag = False

        if flag:
            print(f"obs_remain_cycle_batch is OK !!!!")
        else:
            print(f"obs_remain_cycle_batch is error !!!")
        
        # obs_topic_info のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for topic_idx in range(self.max_topic):
                for channel_idx in range(self.topic_obs_size):
                    if abs(opt_obs_topic_info_batch[batch_idx][topic_idx][channel_idx] - obs_topic_info_batch[batch_idx][topic_idx][channel_idx]) > 0.01:
                        flag = False

        if flag:
            print(f"obs_topic_info_batch is OK !!!!")
        else:
            print(f"obs_topic_info_batch is error !!!")

        # obs_mask のチェック
        flag = True
        for batch_idx in range(self.batch_size):
            for agent_idx in range(self.max_agent):
                for topic_idx in range(self.max_topic):
                    if opt_mask_batch[batch_idx][agent_idx][topic_idx] != mask_batch[batch_idx][agent_idx][topic_idx]:
                        flag = False

        if flag:
            print(f"mask_batch is OK !!!!")
        else:
            print(f"mask_batch is error !!!")
        