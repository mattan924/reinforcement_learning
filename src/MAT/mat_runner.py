from env import Env
from MAT.ma_transformer import MultiAgentTransformer
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
import time as time_module
import datetime
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
    def __init__(self, max_epi_itr, batch_size, device, result_dir, backup_itr, max_agent, max_topic, learning_data_index_path=None, learning_data_index_dir=None, test_data_index_dir=None):

        if not os.path.isdir(result_dir + "model_parameter"):
            sys.exit("結果を格納するディレクトリ" + result_dir + "model_parameter が作成されていません。")

        self.max_epi_itr = max_epi_itr
        self.batch_size = batch_size
        self.device = device
        self.result_dir = result_dir
        self.backup_itr = backup_itr

        # 各種パラメーター
        self.N_action = 9
        self.obs_size = 27
        self.obs_dim = self.obs_size * self.obs_size * 9 + 3
        self.obs_distri_dim = self.obs_size*self.obs_size
        self.obs_info_dim = self.obs_dim - self.obs_distri_dim*9

        self.random_flag = True
        self.test_iter = 10

        self.max_agent = max_agent
        self.max_topic = max_topic

        #  環境のインスタンスの生成
        if learning_data_index_path is not None:
            self.learning_data_index_path = learning_data_index_path
            self.env = Env(learning_data_index_path)
            self.simulation_time = self.env.simulation_time
            self.time_step = self.env.time_step

            if self.simulation_time % self.time_step == 0:
                self.episode_length = int(self.simulation_time / self.time_step)
            else:
                sys.exit("simulation_time が time_step の整数倍になっていません")

            self.policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.obs_info_dim, self.N_action, self.batch_size, max_agent, max_topic, device=self.device, multi=False)
        elif learning_data_index_dir is not None:
            if test_data_index_dir is None:
                sys.exit("評価用の test_data_index_dir を指定して下さい")

            self.learning_data_index_dir = learning_data_index_dir

            train_dir_path = os.path.join(learning_data_index_dir, "*")
            self.train_index_path = glob.glob(train_dir_path)

            test_dir_path = os.path.join(test_data_index_dir, "*")
            self.test_index_path = glob.glob(test_dir_path)

            self.env_list = []
            for idx in range(len(self.train_index_path)):
                self.env_list.append(Env(self.train_index_path[idx]))

            self.test_env_list = []
            for idx in range(len(self.test_index_path)):
                self.test_env_list.append(Env(self.test_index_path[idx]))

            self.simulation_time = self.env_list[0].simulation_time
            self.time_step = self.env_list[0].time_step

            if self.simulation_time % self.time_step == 0:
                self.episode_length = int(self.env_list[0].simulation_time / self.env_list[0].time_step)
            else:
                sys.exit("simulation_time が time_step の整数倍になっていません")

            self.test_buffer = SharedReplayBuffer(self.episode_length, len(self.test_env_list), max_agent, max_topic, self.obs_dim, self.N_action)

            self.policy = TransformerPolicy(self.obs_dim, self.obs_distri_dim, self.obs_info_dim, self.N_action, self.batch_size, max_agent, max_topic, device=self.device, multi=True)
        else:
            sys.exit("学習に使用するデータを指定して下さい")         
        
        self.trainer = MATTrainer(self.policy, self.device)

        self.buffer = SharedReplayBuffer(self.episode_length, self.batch_size, max_agent, max_topic, self.obs_dim, self.N_action)

        self.batch = 0

    
    #  初期準備
    def warmup(self, env, batch, train=True):
        env.reset()

        agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
        obs, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
        #  agent_perm.shape is (max_agent,)
        #  topic_perm.shape is (max_topic,)
        #  obs.shape is (max_agent, max_topic, obs_dim)
        #  mask.shape is (max_agent, max_topic)

        if train:
            self.buffer.obs[0][batch] = obs.reshape(self.max_agent*self.max_topic, self.obs_dim).copy()
            self.buffer.mask[0][batch] = np.bool_(mask.reshape(self.max_agent*self.max_topic).copy())

            self.buffer.agent_perm[0][batch] = agent_perm
            self.buffer.topic_perm[0][batch] = topic_perm

        else:
            self.test_buffer.obs[0][batch] = obs.reshape(self.max_agent*self.max_topic, self.obs_dim).copy()
            self.test_buffer.mask[0][batch] = np.bool_(mask.reshape(self.max_agent*self.max_topic).copy())

            self.test_buffer.agent_perm[0][batch] = agent_perm
            self.test_buffer.topic_perm[0][batch] = topic_perm
                  

    @torch.no_grad()
    def collect(self, step, train=True):
        #  TransformerPolicy を学習用に設定
        self.trainer.prep_rollout()

        if train:
            value, action, action_log_prob = self.trainer.policy.get_actions(self.buffer.obs[step], self.buffer.mask[step], deterministic=False)

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


    def insert(self, batch, obs, mask, rewards, values, actions, action_log_probs, agent_perm, topic_perm, train=True):

        if train:
            self.buffer.insert(batch, obs, mask, actions, action_log_probs, values, rewards, agent_perm, topic_perm)
        else:
            self.test_buffer.insert(batch, obs, mask, actions, action_log_probs, values, rewards, agent_perm, topic_perm)

    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        #  transformer を評価用にセット
        self.trainer.prep_rollout()

        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.obs[-1]), self.buffer.mask[-1])
            
        #  _t2n: tensor から numpy への変換
        next_values = np.array(_t2n(next_values))
        
        self.buffer.compute_returns_batch(next_values, self.trainer.value_normalizer)


    def train(self):
        #  Transformer を train に設定
        self.trainer.prep_training()

        self.trainer.train(self.buffer)

    
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
    
    
    def train_single_env(self, output, transformer_weight, start_epi_itr, load_parameter_path=None):
        timezone_jst = datetime.timezone(datetime.timedelta(hours=9))
        start_process = datetime.datetime.now(timezone_jst)
        print(f"開始時刻: {start_process}")

        if load_parameter_path is not None:
            self.policy.restore(load_parameter_path)
            if start_epi_itr == 0:
                with open(output + ".log", 'w') as f:
                    pass
        else:
            with open(output + ".log", 'w') as f:
                pass

        multi_envs = [Env(self.learning_data_index_path) for _ in range(self.batch_size)]

        # 学習ループ
        for epi_iter in range(start_epi_itr, self.max_epi_itr):
            start_time = time_module.perf_counter()

            #  1エピソード中の reward の保持
            reward_history = [[] for _ in range(self.batch_size)]

            #  環境のリセット            
            for idx in range(self.batch_size):
                env = multi_envs[idx]

                self.warmup(env, idx)
                        
            #  各エピソードにおける時間の推移
            for time in range(0, self.simulation_time, self.time_step):

                step = int(time / self.time_step)

                #  行動と確率分布の取得
                values_batch, actions_batch, action_log_probs_batch = self.collect(step)

                reward_batch = np.zeros((self.batch_size), dtype=np.float32)

                agent_perm_batch = np.zeros((self.batch_size, self.max_agent), dtype=np.int64)
                topic_perm_batch = np.zeros((self.batch_size, self.max_topic), dtype=np.int64)

                obs_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic, self.obs_dim), dtype=np.float32)
                mask_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic), dtype=np.bool)

                # 報酬の受け取り
                for idx in range(self.batch_size):
                    env = multi_envs[idx]
                    reward = env.step(actions_batch[idx][self.buffer.mask[step][idx]], self.buffer.agent_perm[step][idx], self.buffer.topic_perm[step][idx], time)
                    reward_history[idx].append(reward)
                    reward_batch[idx] = -reward

                    #  状態の観測
                    #  ランダムな順にいつか改修
                    agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
                    agent_perm_batch[idx] = agent_perm
                    topic_perm_batch[idx] = topic_perm

                    obs, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
                    obs_batch[idx] = obs
                    mask_batch[idx] = mask
                
                self.insert_batch(obs_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)

            self.compute()

            self.train()
            
            if epi_iter % 1 == 0:
                #  ログの出力
                with open(output + ".log", 'a') as f:
                    reward_average = 0
                    for idx in range(self.batch_size):
                        reward_average += sum(reward_history[idx])/self.batch_size

                    f.write(f"{(epi_iter/self.max_epi_itr)*100}%, {reward_average * -1}\n")

            #  重みパラメータのバックアップ
            if epi_iter % self.backup_itr == 0:
                self.policy.save(self.result_dir + 'model_parameter', transformer_weight, epi_iter)

            end_time = time_module.perf_counter()

            if epi_iter == 0:
                print(f"1 step time = {end_time - start_time}")

                process_time = datetime.timedelta(seconds=(end_time - start_time)*self.max_epi_itr)
                finish_time = start_process + process_time
                print(f"終了予定時刻: {finish_time}")

        #  最適解の取り出し
        df_index = pd.read_csv(self.learning_data_index_path, index_col=0)
        opt = df_index.at['data', 'opt']

        #  近傍サーバを選択した際の遅延を計算
        reward_history = [[] for _ in range(self.batch_size)]

        #  環境のリセット            
        for idx in range(self.batch_size):
            env = multi_envs[idx]

            self.warmup(env, idx)

        for time in range(0, self.simulation_time, self.time_step):

            step = int(time / self.time_step)

            #  行動と確率分布の取得
            values_batch, actions_batch, action_log_probs_batch = self.collect(step)

            actions_batch = np.zeros((self.batch_size, self.max_agent*self.max_topic, 1))

            for idx in range(self.batch_size):
                actions = self.env.get_near_action(self.buffer.agent_perm[step][idx], self.buffer.topic_perm[step][idx])
                actions_batch[idx] = actions

            reward_batch = np.zeros((self.batch_size), dtype=np.float32)

            agent_perm_batch = np.zeros((self.batch_size, self.max_agent), dtype=np.int64)
            topic_perm_batch = np.zeros((self.batch_size, self.max_topic), dtype=np.int64)

            obs_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic, self.obs_dim), dtype=np.float32)
            mask_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic), dtype=np.bool)

            # 報酬の受け取り
            for idx in range(self.batch_size):
                env = multi_envs[idx]
                reward = env.step(actions_batch[idx][self.buffer.mask[step][idx]], self.buffer.agent_perm[step][idx], self.buffer.topic_perm[step][idx], time)
                reward_history[idx].append(reward)
                reward_batch[idx] = -reward

                #  状態の観測
                #  ランダムな順にいつか改修
                agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
                agent_perm_batch[idx] = agent_perm
                topic_perm_batch[idx] = topic_perm

                obs, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
                obs_batch[idx] = obs
                mask_batch[idx] = mask
                
            self.insert_batch(obs_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)

        reward_average = 0
        for idx in range(self.batch_size):
            reward_average += sum(reward_history[idx])/self.batch_size

        print(f"reward_average = {reward_average}")

        train_curve = read_train_curve(output + ".log")

        #  学習曲線の描画
        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        wind.grid()
        wind.plot(train_curve, linewidth=1, label='')
        wind.axhline(y=-opt, c='r', label='opt')
        wind.axhline(y=-reward_average, c='g', label='nearest')
        fig.savefig(output + ".png")

        #  重みパラメータの保存
        self.policy.save(self.result_dir + 'model_parameter', transformer_weight, epi_iter+1)


    def train_multi_env(self, output, transformer_weight, start_epi_itr, load_parameter_path=None):
        timezone_jst = datetime.timezone(datetime.timedelta(hours=9))
        start_process = datetime.datetime.now(timezone_jst)
        print(f"開始時刻: {start_process}")

        if load_parameter_path is not None:
            self.policy.restore(load_parameter_path)
            if start_epi_itr == 0:
                with open(output + ".log", 'w') as f:
                    pass
                
                for idx in range(len(self.test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'w') as f:
                        pass
        else:
            with open(output + ".log", 'w') as f:
                pass

            for idx in range(len(self.test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'w') as f:
                        pass


        # 学習ループ
        for epi_iter in range(start_epi_itr, self.max_epi_itr):
            start_time = time_module.perf_counter()

            #  環境のリセット
            self.env_list_shuffle = random.sample(self.env_list, self.batch_size)

            #  1エピソード中の reward の保持
            reward_history = [[] for _ in range(self.batch_size)]

            warmup_start = time_module.perf_counter()

            #  環境のリセット            
            for idx in range(self.batch_size):
                env = self.env_list_shuffle[idx]

                self.warmup(env, idx)

            warmup_end = time_module.perf_counter()
                        
            #  各エピソードにおける時間の推移
            for time in range(0, self.simulation_time, self.time_step):

                step = int(time / self.time_step)

                collect_start = time_module.perf_counter()

                #  行動と確率分布の取得
                values_batch, actions_batch, action_log_probs_batch = self.collect(step)

                collect_end = time_module.perf_counter()

                reward_batch = np.zeros((self.batch_size), dtype=np.float32)

                agent_perm_batch = np.zeros((self.batch_size, self.max_agent), dtype=np.int64)
                topic_perm_batch = np.zeros((self.batch_size, self.max_topic), dtype=np.int64)

                obs_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic, self.obs_dim), dtype=np.float32)
                mask_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic), dtype=np.bool)

                # 報酬の受け取り
                for idx in range(self.batch_size):
                    step_start = time_module.perf_counter()

                    env = self.env_list_shuffle[idx]
                    reward = env.step(actions_batch[idx], self.buffer.agent_perm[step][idx], self.buffer.topic_perm[step][idx], time)
                    reward_history[idx].append(reward)
                    reward_batch[idx] = -reward

                    step_end = time_module.perf_counter()

                    observ_start = time_module.perf_counter()

                    #  状態の観測
                    #  ランダムな順にいつか改修
                    agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
                    agent_perm_batch[idx] = agent_perm
                    topic_perm_batch[idx] = topic_perm

                    obs, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
                    obs_batch[idx] = obs
                    mask_batch[idx] = mask

                    observ_end = time_module.perf_counter()
                
                insert_start = time_module.perf_counter()
                
                self.insert_batch(obs_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)

                insert_end = time_module.perf_counter()

            compute_start = time_module.perf_counter()

            self.compute()

            compute_end = time_module.perf_counter()

            train_start = time_module.perf_counter()

            self.train()

            train_end = time_module.perf_counter()
            
            if epi_iter % 1 == 0:
                #  ログの出力
                with open(output + ".log", 'a') as f:
                    reward_average = 0
                    for idx in range(self.batch_size):
                        reward_average += sum(reward_history[idx])/self.batch_size

                    f.write(f"{(epi_iter/self.max_epi_itr)*100}%, {reward_average * -1}\n")

            if epi_iter % self.test_iter == 0 or (epi_iter+1) == self.max_epi_itr:
                test_start = time_module.perf_counter()

                for idx in range(len(self.test_env_list)):
                    test_env = self.test_env_list[idx]

                    self.warmup(test_env, idx, train=False)

                #  各エピソードにおける時間の推移
                reward_history_test = [[] for _ in range(len(self.test_env_list))]

                for time in range(0, test_env.simulation_time, test_env.time_step):
                    step = int(time / test_env.time_step)

                    #  行動と確率分布の取得
                    values_batch, actions_batch, action_log_probs_batch = self.collect(step, train=False)

                    reward_batch = np.zeros((len(self.test_env_list)), dtype=np.float32)

                    agent_perm_batch = np.zeros((len(self.test_env_list), self.max_agent), dtype=np.int64)
                    topic_perm_batch = np.zeros((len(self.test_env_list), self.max_topic), dtype=np.int64)

                    obs_batch = np.zeros((len(self.test_env_list), self.max_agent, self.max_topic, self.obs_dim), dtype=np.float32)
                    mask_batch = np.zeros((len(self.test_env_list), self.max_agent, self.max_topic), dtype=np.bool)

                    # 報酬の受け取り
                    for idx in range(len(self.test_env_list)):
                        test_env = self.test_env_list[idx]
                        reward = test_env.step(actions_batch[idx], self.test_buffer.agent_perm[step][idx], self.test_buffer.topic_perm[step][idx], time)
                        reward_history_test[idx].append(reward)
                        reward_batch[idx] = -reward

                        #  状態の観測
                        #  ランダムな順にいつか改修
                        agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
                        agent_perm_batch[idx] = agent_perm
                        topic_perm_batch[idx] = topic_perm

                        obs, mask = test_env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
                        obs_batch[idx] = obs
                        mask_batch[idx] = mask

                    self.insert_batch(obs_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch, train=False)

                for idx in range(len(self.test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'a') as f:
                        f.write(f"{(epi_iter/self.max_epi_itr)*100}%, {-sum(reward_history_test[idx])}\n")

                test_end = time_module.perf_counter()

            #  重みパラメータのバックアップ
            if epi_iter % self.backup_itr == 0:
                self.policy.save(self.result_dir + 'model_parameter', transformer_weight, epi_iter)

            end_time = time_module.perf_counter()

            if epi_iter == 0:
                #print(f"1 step time = {end_time - start_time}")
                print(f"1 step time = {train_end - start_time}")

                print(f"warmup time = {warmup_end - warmup_start}")
                print(f"collect time = {(collect_end - collect_start)*60}")
                print(f"step time = {(step_end - step_start)*16*60}")
                print(f"observ time = {(observ_end - observ_start)*16*60}")
                print(f"insert time = {(insert_end - insert_start)*60}")
                print(f"compute time = {compute_end - compute_start}")
                print(f"train time = {train_end - train_start}")

                process_time = datetime.timedelta(seconds=(end_time - start_time - (test_end -test_start))*self.max_epi_itr + (test_end - test_start)*(self.max_epi_itr / self.test_iter))
                finish_time = start_process + process_time
                print(f"終了予定時刻: {finish_time}")

        #  重みパラメータの保存
        self.policy.save(self.result_dir + 'model_parameter', transformer_weight, epi_iter+1)


    def train_multi_env_debug(self, output, transformer_weight, start_epi_itr, load_parameter_path=None):
        timezone_jst = datetime.timezone(datetime.timedelta(hours=9))
        start_process = datetime.datetime.now(timezone_jst)
        print(f"開始時刻: {start_process}")

        start_time = time_module.perf_counter()


        num_used_env = 2
        self.env_list = []
        self.test_env_list = []
        for i in range(num_used_env):
            index_path = self.train_index_path[i]
            self.test_env_list.append(Env(index_path))
            for _ in range(int(self.batch_size/num_used_env)):
                self.env_list.append(Env(index_path))

        self.test_buffer = SharedReplayBuffer(self.episode_length, len(self.test_env_list), self.max_agent, self.max_topic, self.obs_dim, self.N_action)


        if load_parameter_path is not None:
            self.policy.restore(load_parameter_path)
            if start_epi_itr == 0:
                with open(output + ".log", 'w') as f:
                    pass
                
                for idx in range(len(self.test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'w') as f:
                        pass
        else:
            with open(output + ".log", 'w') as f:
                pass

            for idx in range(len(self.test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'w') as f:
                        pass


        # 学習ループ
        for epi_iter in range(start_epi_itr, self.max_epi_itr):

            #  環境のリセット
            self.env_list_shuffle = random.sample(self.env_list, self.batch_size)

            #  1エピソード中の reward の保持
            reward_history = [[] for _ in range(self.batch_size)]

            #  環境のリセット            
            for idx in range(self.batch_size):
                env = self.env_list_shuffle[idx]

                self.warmup(env, idx)
                        
            #  各エピソードにおける時間の推移
            for time in range(0, self.simulation_time, self.time_step):

                step = int(time / self.time_step)

                #  行動と確率分布の取得
                values_batch, actions_batch, action_log_probs_batch = self.collect(step)

                reward_batch = np.zeros((self.batch_size), dtype=np.float32)

                agent_perm_batch = np.zeros((self.batch_size, self.max_agent), dtype=np.int64)
                topic_perm_batch = np.zeros((self.batch_size, self.max_topic), dtype=np.int64)

                obs_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic, self.obs_dim), dtype=np.float32)
                mask_batch = np.zeros((self.batch_size, self.max_agent, self.max_topic), dtype=np.bool)

                # 報酬の受け取り
                for idx in range(self.batch_size):
                    env = self.env_list_shuffle[idx]
                    reward = env.step(actions_batch[idx], self.buffer.agent_perm[step][idx], self.buffer.topic_perm[step][idx], time)
                    reward_history[idx].append(reward)
                    reward_batch[idx] = -reward

                    #  状態の観測
                    #  ランダムな順にいつか改修
                    agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
                    agent_perm_batch[idx] = agent_perm
                    topic_perm_batch[idx] = topic_perm

                    obs, mask = env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
                    obs_batch[idx] = obs
                    mask_batch[idx] = mask

                self.insert_batch(obs_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)

            self.compute()

            self.train()
            
            if epi_iter % 1 == 0:
                #  ログの出力
                with open(output + ".log", 'a') as f:
                    reward_average = 0
                    for idx in range(self.batch_size):
                        reward_average += sum(reward_history[idx])/self.batch_size

                    f.write(f"{(epi_iter/self.max_epi_itr)*100}%, {reward_average * -1}\n")

            if epi_iter % self.test_iter == 0 or (epi_iter+1) == self.max_epi_itr:
                test_start = time_module.perf_counter()

                for idx in range(len(self.test_env_list)):
                    test_env = self.test_env_list[idx]

                    self.warmup(test_env, idx, train=False)

                #  各エピソードにおける時間の推移
                reward_history_test = [[] for _ in range(len(self.test_env_list))]

                for time in range(0, test_env.simulation_time, test_env.time_step):
                    step = int(time / test_env.time_step)

                    #  行動と確率分布の取得
                    values_batch, actions_batch, action_log_probs_batch = self.collect(step, train=False)

                    reward_batch = np.zeros((len(self.test_env_list)), dtype=np.float32)

                    agent_perm_batch = np.zeros((len(self.test_env_list), self.max_agent), dtype=np.int64)
                    topic_perm_batch = np.zeros((len(self.test_env_list), self.max_topic), dtype=np.int64)

                    obs_batch = np.zeros((len(self.test_env_list), self.max_agent, self.max_topic, self.obs_dim), dtype=np.float32)
                    mask_batch = np.zeros((len(self.test_env_list), self.max_agent, self.max_topic), dtype=np.bool)

                    # 報酬の受け取り
                    for idx in range(len(self.test_env_list)):
                        test_env = self.test_env_list[idx]
                        reward = test_env.step(actions_batch[idx], self.test_buffer.agent_perm[step][idx], self.test_buffer.topic_perm[step][idx], time)
                        reward_history_test[idx].append(reward)
                        reward_batch[idx] = -reward

                        #  状態の観測
                        #  ランダムな順にいつか改修
                        agent_perm, topic_perm = self.get_perm(random_flag=self.random_flag)
                        agent_perm_batch[idx] = agent_perm
                        topic_perm_batch[idx] = topic_perm

                        obs, mask = test_env.get_observation_mat(agent_perm, topic_perm, self.obs_size)
                        obs_batch[idx] = obs
                        mask_batch[idx] = mask

                    self.insert_batch(obs_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch, train=False)

                for idx in range(len(self.test_env_list)):
                    with open(output + "_test" + str(idx) + ".log", 'a') as f:
                        f.write(f"{(epi_iter/self.max_epi_itr)*100}%, {-sum(reward_history_test[idx])}\n")

                test_end = time_module.perf_counter()

            #  重みパラメータのバックアップ
            if epi_iter % self.backup_itr == 0:
                self.policy.save(self.result_dir + 'model_parameter', transformer_weight, epi_iter)

            end_time = time_module.perf_counter()

            if epi_iter == 0:
                process_time = datetime.timedelta(seconds=(end_time - start_time - (test_end - test_start))*self.max_epi_itr + (test_end - test_start)*(self.max_epi_itr / self.test_iter))
                finish_time = start_process + process_time
                print(f"終了予定時刻: {finish_time}")

        #  重みパラメータの保存
        self.policy.save(self.result_dir + 'model_parameter', transformer_weight, epi_iter+1)
