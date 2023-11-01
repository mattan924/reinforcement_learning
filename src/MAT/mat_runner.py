from env import Env
from MAT.transformer_policy import TransformerPolicy
from MAT.mat_trainer import MATTrainer
from MAT.shared_buffer import SharedReplayBuffer
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import glob
import random
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
    def __init__(self, batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd, reward_scaling, device, max_agent, max_topic):

        # ハイパーパラメーター
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.obs_size = obs_size
        self.n_block = n_block
        self.n_embd = n_embd
        self.reward_scaling = reward_scaling

        # 各種パラメーター
        self.device = device
        self.max_agent = max_agent
        self.max_topic = max_topic
        self.N_action = 9
        self.obs_dim = self.obs_size * self.obs_size * 9 + 3
        self.obs_distri_dim = self.obs_size*self.obs_size
        self.obs_info_dim = self.obs_dim - self.obs_distri_dim*9

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

    
    #  初期準備
    def warmup(self, buffer, env, batch):
        env.reset()

        agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
        obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)

        buffer.obs_posi[0][batch] = obs_posi
        buffer.obs_publisher[0][batch] = obs_publisher
        buffer.obs_subscriber[0][batch] = obs_subscriber
        buffer.obs_distribution[0][batch] = obs_distribution
        buffer.obs_topic_used_storage[0][batch] = obs_topic_used_storage
        buffer.obs_storage[0][batch] = obs_storage
        buffer.obs_cpu_cycle[0][batch] = obs_cpu_cycle
        buffer.obs_topic_num_used[0][batch] = obs_topic_num_used
        buffer.obs_num_used[0][batch] = obs_num_used
        buffer.obs_topic_info[0][batch] = obs_topic_info
        buffer.mask[0][batch] = np.bool_(mask.reshape(self.max_agent*self.max_topic))

        buffer.agent_perm[0][batch] = agent_perm
        buffer.topic_perm[0][batch] = topic_perm


    @torch.no_grad()
    def collect(self, trainer, buffer, step, train=True):
        #  TransformerPolicy を学習用に設定
        trainer.prep_rollout()

        obs_posi = buffer.obs_posi[step]
        batch_size = buffer.obs_posi.shape[1]
        obs_client = np.zeros((batch_size, self.max_topic, self.obs_distri_dim*3), dtype=np.float32)
        obs_client[:, :, :self.obs_distri_dim] = buffer.obs_publisher[step]
        obs_client[:, :, self.obs_distri_dim:self.obs_distri_dim*2] = buffer.obs_subscriber[step]
        obs_client[:, :, self.obs_distri_dim*2:self.obs_distri_dim*3] = buffer.obs_distribution[step][:, np.newaxis]
            
        obs_edge = np.zeros((batch_size, self.max_topic, self.obs_distri_dim*5), dtype=np.float32)
        obs_edge[:, :, :self.obs_distri_dim] = buffer.obs_topic_used_storage[step]
        obs_edge[:, :, self.obs_distri_dim:self.obs_distri_dim*2] = buffer.obs_storage[step][:, np.newaxis]
        obs_edge[:, :, self.obs_distri_dim*2:self.obs_distri_dim*3] = buffer.obs_cpu_cycle[step][:, np.newaxis]
        obs_edge[:, :, self.obs_distri_dim*3:self.obs_distri_dim*4] = buffer.obs_topic_num_used[step]
        obs_edge[:, :, self.obs_distri_dim*4:self.obs_distri_dim*5] = buffer.obs_num_used[step][:, np.newaxis]
            
        obs_topic_info = buffer.obs_topic_info[step]

        if train==True:
            value, action, action_log_prob = trainer.policy.get_actions(obs_posi, obs_client, obs_edge, obs_topic_info, buffer.mask[step], deterministic=False)
        else:
            value, action, action_log_prob = trainer.policy.get_actions(obs_posi, obs_client, obs_edge, obs_topic_info, buffer.mask[step], deterministic=True)

        # _t2n: tensor → numpy
        values = np.array(_t2n(value))
        actions = np.array(_t2n(action))
        action_log_probs = np.array(_t2n(action_log_prob))

        return values, actions, action_log_probs
    

    def insert_batch(self, buffer, obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask, rewards, values, actions, action_log_probs, agent_perm, topic_perm):
            
        buffer.insert_batch(obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask, actions, action_log_probs, values, rewards, agent_perm, topic_perm)

    
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
            
        obs_edge = np.zeros((self.batch_size, self.max_topic, self.obs_distri_dim*5), dtype=np.float32)
        obs_edge[:, :, :self.obs_distri_dim] = buffer.obs_topic_used_storage[-1]
        obs_edge[:, :, self.obs_distri_dim:self.obs_distri_dim*2] = buffer.obs_storage[-1][:, np.newaxis]
        obs_edge[:, :, self.obs_distri_dim*2:self.obs_distri_dim*3] = buffer.obs_cpu_cycle[-1][:, np.newaxis]
        obs_edge[:, :, self.obs_distri_dim*3:self.obs_distri_dim*4] = buffer.obs_topic_num_used[-1]
        obs_edge[:, :, self.obs_distri_dim*4:self.obs_distri_dim*5] = buffer.obs_num_used[-1][:, np.newaxis]
            
        obs_topic_info = buffer.obs_topic_info[-1]

        next_values = trainer.policy.get_values(obs_posi, obs_client, obs_edge, obs_topic_info, buffer.mask[-1])
            
        #  _t2n: tensor から numpy への変換
        next_values = np.array(_t2n(next_values))
        
        buffer.compute_returns_batch(next_values, trainer.value_normalizer)


    def train(self, trainer, buffer):
        #  Transformer を train に設定
        trainer.prep_training()

        trainer.train(buffer)


    def episode_loop(self, simulation_time, time_step, trainer, buffer, batch_size, env_list, reward_history, train=True):
        #  各エピソードにおける時間の推移
        for time in range(0, simulation_time, time_step):
            step = int(time / time_step)

            #  行動と確率分布の取得
            if train == True:
                values_batch, actions_batch, action_log_probs_batch = self.collect(trainer, buffer, step, train=True)
            else:
                values_batch, actions_batch, action_log_probs_batch = self.collect(trainer, buffer, step, train=False)

            reward_batch = np.zeros((batch_size), dtype=np.float32)

            agent_perm_batch = np.zeros((batch_size, self.max_agent), dtype=np.int64)
            topic_perm_batch = np.zeros((batch_size, self.max_topic), dtype=np.int64)

            obs_posi_batch = np.zeros((batch_size, self.max_agent, self.obs_distri_dim), dtype=np.float32)
            obs_publisher_batch = np.zeros((batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_subscriber_batch = np.zeros((batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_distribution_batch = np.zeros((batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_topic_used_storage_batch = np.zeros((batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_storage_batch = np.zeros((batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_cpu_cycle_batch = np.zeros((batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_topic_num_used_batch = np.zeros((batch_size, self.max_topic, self.obs_distri_dim), dtype=np.float32)
            obs_num_used_batch = np.zeros((batch_size, self.obs_distri_dim), dtype=np.float32)
            obs_topic_info_batch = np.zeros((batch_size, self.max_topic, 3), dtype=np.float32)
            mask_batch = np.zeros((batch_size, self.max_agent, self.max_topic), dtype=np.bool)

            # 報酬の受け取り
            for idx in range(batch_size):
                env = env_list[idx]
                reward = env.step(actions_batch[idx][buffer.mask[step][idx]], buffer.agent_perm[step][idx], buffer.topic_perm[step][idx], time)
                reward_history[idx].append(reward)
                if self.reward_scaling == True:
                    reward_batch[idx] = (-reward / 200) + 1
                else:
                    reward_batch[idx] = -reward

                #  状態の観測
                #  ランダムな順にいつか改修
                agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
                agent_perm_batch[idx] = agent_perm
                topic_perm_batch[idx] = topic_perm

                obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
                obs_posi_batch[idx] = obs_posi
                obs_publisher_batch[idx] = obs_publisher
                obs_subscriber_batch[idx] = obs_subscriber
                obs_distribution_batch[idx] = obs_distribution
                obs_topic_used_storage_batch[idx] = obs_topic_used_storage
                obs_storage_batch[idx] = obs_storage
                obs_cpu_cycle_batch[idx] = obs_cpu_cycle
                obs_topic_num_used_batch[idx] = obs_topic_num_used
                obs_num_used_batch[idx] = obs_num_used
                obs_topic_info_batch[idx] = obs_topic_info
                mask_batch[idx] = mask
                
            self.insert_batch(buffer, obs_posi_batch, obs_publisher_batch, obs_subscriber_batch, obs_distribution_batch, obs_topic_used_storage_batch, obs_storage_batch, obs_cpu_cycle_batch, obs_topic_num_used_batch, obs_num_used_batch, obs_topic_info_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)
    

    def cal_nearest_server_reward(self, index_path):
        nearest_reward = 0

        env = Env(index_path)
        simulation_time = env.simulation_time
        time_step = env.time_step

        agent_perm, topic_perm = self.get_perm(random_flag=False)

        for time in range(0, simulation_time, time_step):
            obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, obs_size=self.obs_size)
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

        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd, device=self.device, multi=False)

        trainer = MATTrainer(policy, self.ppo_epoch, self.device)

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


    def train_multi_env(self, start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, result_dir, output, transformer_weight, backup_itr, load_parameter_path=None):
        test_iter = 10

        if test_data_index_dir is None:
            sys.exit("評価用の test_data_index_dir を指定して下さい")

        train_dir_path = os.path.join(learning_data_index_dir, "*")
        train_index_path = natsorted(glob.glob(train_dir_path))

        test_dir_path = os.path.join(test_data_index_dir, "*")
        test_index_path = natsorted(glob.glob(test_dir_path))

        env_list = []
        for idx in range(len(train_index_path)):
            env_list.append(Env(train_index_path[idx]))

        test_env_list = []
        for idx in range(len(test_index_path)):
            test_env_list.append(Env(test_index_path[idx]))

        simulation_time = env_list[0].simulation_time
        time_step = env_list[0].time_step

        episode_length = int(simulation_time / time_step)
        for idx in range(len(env_list)):
            if env_list[idx].simulation_time % env_list[idx].time_step != 0:
                sys.exit("simulation_time が time_step の整数倍になっていません")
            elif env_list[idx].simulation_time != simulation_time or env_list[idx].time_step != time_step:
                sys.exit("データセット内に異なる simulation_time または time_step が含まれています。")
            
        for idx in range(len(test_env_list)):
            if test_env_list[idx].simulation_time % test_env_list[idx].time_step != 0:
                sys.exit("simulation_time が time_step の整数倍になっていません")
            elif test_env_list[idx].simulation_time != simulation_time or test_env_list[idx].time_step != time_step:
                sys.exit("テストデータセット内に異なる simulation_time または time_step が含まれています。")

        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd, device=self.device, multi=True)
        trainer = MATTrainer(policy, self.ppo_epoch, self.device)

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
            env_list_shuffle = random.sample(env_list, self.batch_size)

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

                self.episode_loop(simulation_time, time_step, trainer, test_buffer, len(test_env_list), test_env_list, reward_history_test, train=False)

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

                print(f"warmup time = {warmup_end - warmup_start}")
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

        policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.obs_info_dim, self.N_action, self.batch_size, self.max_agent, self.max_topic, self.lr, self.eps, self.weight_decay, self.n_block, self.n_embd, device=self.device, multi=False)

        trainer = MATTrainer(policy, self.ppo_epoch, self.device)

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