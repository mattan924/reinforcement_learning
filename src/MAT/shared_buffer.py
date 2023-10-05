import torch
import numpy as np
import torch.nn.functional as F
import time


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):

    def __init__(self, episode_length, batch_size, num_agents, num_topic, obs_dim, act_dim):
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self._use_gae = True
        self._use_popart = False
        self._use_valuenorm = True
        self._use_proper_time_limits = False
        self.num_agents = num_agents
        self.num_topic = num_topic

        self.obs_dim = obs_dim
        self.obs_size2 = int((self.obs_dim-3)/9)
        self.act_dim = act_dim

        self.obs_posi = np.zeros((self.episode_length + 1, self.batch_size, self.num_agents, self.obs_size2), dtype=np.float32)
        self.obs_publisher = np.zeros((self.episode_length + 1, self.batch_size, self.num_topic, self.obs_size2), dtype=np.float32)
        self.obs_subscriber = np.zeros((self.episode_length + 1, self.batch_size, self.num_topic, self.obs_size2), dtype=np.float32)
        self.obs_distribution = np.zeros((self.episode_length + 1, self.batch_size, self.obs_size2), dtype=np.float32)
        self.obs_topic_used_storage = np.zeros((self.episode_length + 1, self.batch_size, self.num_topic, self.obs_size2), dtype=np.float32)
        self.obs_storage = np.zeros((self.episode_length + 1, self.batch_size, self.obs_size2), dtype=np.float32)
        self.obs_cpu_cycle = np.zeros((self.episode_length + 1, self.batch_size, self.obs_size2), dtype=np.float32)
        self.obs_topic_num_used = np.zeros((self.episode_length + 1, self.batch_size, self.num_topic, self.obs_size2), dtype=np.float32)
        self.obs_num_used = np.zeros((self.episode_length + 1, self.batch_size, self.obs_size2), dtype=np.float32)
        self.obs_topic_info = np.zeros((self.episode_length + 1, self.batch_size, self.num_topic, 3), dtype=np.float32)
        self.mask = np.zeros((self.episode_length + 1, self.batch_size, self.num_agents*self.num_topic), dtype=np.bool)

        self.value_preds = np.zeros((self.episode_length + 1, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros((self.episode_length, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)
        
        self.actions = np.ones((self.episode_length, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)*-1
        self.action_log_probs = np.zeros((self.episode_length, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)

        self.agent_perm = np.zeros((self.episode_length + 1, self.batch_size, self.num_agents), dtype=np.int64)
        self.topic_perm = np.zeros((self.episode_length + 1, self.batch_size, self.num_topic), dtype=np.int64)

        self.step = 0
        

    #  データを挿入する
    def insert_batch(self, obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask, actions, action_log_probs, value_preds, rewards, agent_perm, topic_perm):

        self.obs_posi[self.step + 1] = obs_posi
        self.obs_publisher[self.step + 1] = obs_publisher
        self.obs_subscriber[self.step + 1] = obs_subscriber
        self.obs_distribution[self.step + 1] = obs_distribution
        self.obs_topic_used_storage[self.step + 1] = obs_topic_used_storage
        self.obs_storage[self.step + 1] = obs_storage
        self.obs_cpu_cycle[self.step + 1] = obs_cpu_cycle
        self.obs_topic_num_used[self.step + 1] = obs_topic_num_used
        self.obs_num_used[self.step + 1] = obs_num_used
        self.obs_topic_info[self.step + 1] = obs_topic_info
        self.mask[self.step + 1] = np.bool_(mask.reshape(self.batch_size, self.num_agents*self.num_topic))
        self.actions[self.step][self.mask[self.step]] = actions[self.mask[self.step]]
        self.action_log_probs[self.step][self.mask[self.step]] = action_log_probs[self.mask[self.step]]
        self.value_preds[self.step][self.mask[self.step]] = value_preds.reshape(-1, 1)
        for batch in range(self.batch_size):
            self.rewards[self.step][batch][self.mask[self.step][batch]] = rewards[batch]
        self.agent_perm[self.step + 1] = agent_perm
        self.topic_perm[self.step + 1] = topic_perm

        self.step = (self.step + 1) % self.episode_length

    
    def compute_returns_batch(self, next_value, value_normalizer=None):
        """
        報酬の割引和として、または GAE を使用してリターンを計算します。
        :param next_value: (np.ndarray) 最後のエピソードステップの次のステップの値予測。
        :param value_normalizer: (PopArt) Noneでない場合、PopArt値のノーマライザインスタンス。
        """

        self.value_preds[-1][self.mask[-1]] = next_value

        gae = 0

        for step in reversed(range(self.episode_length)):
            delta = self.rewards[step][self.mask[step]] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1][self.mask[step + 1]]) - value_normalizer.denormalize(self.value_preds[step][self.mask[step]])
            gae = delta + self.gamma * self.gae_lambda * gae

            self.advantages[step][self.mask[step]] = gae
            self.returns[step][self.mask[step]] = gae + value_normalizer.denormalize(self.value_preds[step][self.mask[step]])


    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        MLP ポリシーの学習データを取得．
        param advantages: (np.ndarray) アドバンテージ推定値．
        param num_mini_batch: （int） バッチを分割するミニバッチの数．
        param mini_batch_size: （int） 各ミニバッチ内のサンプル数．
        """

        #  mini_batch_size is None
        #  num_mini_batch is 1
        #  batch_size is n_rollout_threads * episode_length

        if mini_batch_size is None:
            
            mini_batch_size = self.batch_size*self.episode_length // num_mini_batch

        rand = torch.randperm(self.batch_size*self.episode_length).numpy()
        indices = rand[:mini_batch_size]
        
        obs_posi = self.obs_posi[:-1]

        obs_client = np.zeros((self.episode_length, self.batch_size, self.num_topic, self.obs_size2*3), dtype=np.float32)
        obs_client[:, :, :, :self.obs_size2] = self.obs_publisher[:-1]
        obs_client[:, :, :, self.obs_size2:self.obs_size2*2] = self.obs_subscriber[:-1]
        obs_client[:, :, :, self.obs_size2*2:self.obs_size2*3] = self.obs_distribution[:-1][:, :, np.newaxis]

        obs_edge = np.zeros((self.episode_length, self.batch_size, self.num_topic, self.obs_size2*5), dtype=np.float32)
        obs_edge[:, :, :, :self.obs_size2] = self.obs_topic_used_storage[:-1]
        obs_edge[:, :, :, self.obs_size2:self.obs_size2*2] = self.obs_storage[:-1][:, :, np.newaxis]
        obs_edge[:, :, :, self.obs_size2*2:self.obs_size2*3] = self.obs_cpu_cycle[:-1][:, :, np.newaxis]
        obs_edge[:, :, :, self.obs_size2*3:self.obs_size2*4] = self.obs_topic_num_used[:-1]
        obs_edge[:, :, :, self.obs_size2*4:self.obs_size2*5] = self.obs_num_used[:-1][:, :, np.newaxis]

        obs_topic_info = self.obs_topic_info[:-1]

        obs_posi = obs_posi.reshape(-1, *obs_posi.shape[2:])
        obs_client = obs_client.reshape(-1, *obs_client.shape[2:])
        obs_edge = obs_edge.reshape(-1, *obs_edge.shape[2:])
        obs_topic_info = obs_topic_info.reshape(-1, *obs_topic_info.shape[2:])
        mask = self.mask[:-1].reshape(-1, *self.mask.shape[2:])

        actions = self.actions.reshape(-1, *self.actions.shape[2:])

        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])

        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])

        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])

        advantages = advantages.reshape(-1, *advantages.shape[2:])

        obs_posi_batch = obs_posi.reshape(-1, *obs_posi.shape[2:])
        obs_client_batch = obs_client.reshape(-1, *obs_client.shape[2:])
        obs_edge_batch = obs_edge.reshape(-1, *obs_edge.shape[2:])
        obs_topic_info_batch = obs_topic_info.reshape(-1, *obs_topic_info.shape[2:])
        mask_batch = mask.reshape(-1, *mask.shape[1:])
        actions_batch = actions.reshape(-1, *actions.shape[2:])

        value_preds_batch = value_preds.reshape(-1, *value_preds.shape[2:])
        return_batch = returns.reshape(-1, *returns.shape[2:])
        old_action_log_probs_batch = action_log_probs.reshape(-1, *action_log_probs.shape[2:])

        if advantages is None:
            adv_targ = None
        else:
            adv_targ = advantages.reshape(-1, *advantages.shape[2:])

        return obs_posi_batch, obs_client_batch, obs_edge_batch, obs_topic_info_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, mask_batch