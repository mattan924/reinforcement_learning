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
    """
    学習データを格納するバッファ。
    param args: (argparse.Namespace) 関連するモデル，ポリシー，環境情報を含む引数．
    param num_agents: (int) 環境内のエージェント数。
    param obs_space: (gym.Space) エージェントの観測空間．
    param cent_obs_space: (gym.Space) エージェントの集中観測空間．
    param act_space: (gym.Space) エージェントの行動空間．
    """

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
        self.act_dim = act_dim

        self.obs = np.zeros((self.episode_length + 1, self.batch_size, self.num_agents*self.num_topic, self.obs_dim), dtype=np.float32)
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
        

    def insert(self, batch, obs, mask, actions, action_log_probs, value_preds, rewards, agent_perm, topic_perm):
        """
        バッファにデータを挿入します。
        param obs: (np.ndarray) ローカルエージェントのオブザベーション。
        :param actions:(np.ndarray) エージェントが取った行動。
        param action_log_probs:(np.ndarray) エージェントが取った行動のログ確率。
        param value_preds: (np.ndarray) 各ステップにおける値関数の予測値．
        param rewards: (np.ndarray) 各ステップで収集した報酬。
        """

        self.obs[self.step + 1][batch] = obs.reshape(self.num_agents*self.num_topic, self.obs_dim).copy()
        self.mask[self.step + 1][batch] = np.bool_(mask.reshape(self.num_agents*self.num_topic).copy())
        self.actions[self.step][batch][self.mask[self.step][batch]] = actions.copy()
        self.action_log_probs[self.step][batch][self.mask[self.step][batch]] = action_log_probs.copy()
        self.value_preds[self.step][batch][self.mask[self.step][batch]] = value_preds.copy()
        self.rewards[self.step][batch][self.mask[self.step][batch]] = rewards.copy()
        self.agent_perm[self.step + 1][batch] = agent_perm.copy()
        self.topic_perm[self.step + 1][batch] = topic_perm.copy()

        self.step = (self.step + 1) % self.episode_length

    
    def insert_batch(self, obs, mask, actions, action_log_probs, value_preds, rewards, agent_perm, topic_perm):
        """
        バッファにデータを挿入します。
        param obs: (np.ndarray) ローカルエージェントのオブザベーション。
        :param actions:(np.ndarray) エージェントが取った行動。
        param action_log_probs:(np.ndarray) エージェントが取った行動のログ確率。
        param value_preds: (np.ndarray) 各ステップにおける値関数の予測値．
        param rewards: (np.ndarray) 各ステップで収集した報酬。
        """

        self.obs[self.step + 1] = obs.reshape(self.batch_size, self.num_agents*self.num_topic, self.obs_dim).copy()
        self.mask[self.step + 1] = np.bool_(mask.reshape(self.batch_size, self.num_agents*self.num_topic).copy())
        self.actions[self.step][self.mask[self.step]] = actions.reshape(-1, 1).copy()
        self.action_log_probs[self.step][self.mask[self.step]] = action_log_probs.reshape(-1, 1).copy()
        self.value_preds[self.step][self.mask[self.step]] = value_preds.reshape(-1, 1).copy()
        for batch in range(self.batch_size):
            self.rewards[self.step][batch][self.mask[self.step][batch]] = rewards[batch].copy()
        self.agent_perm[self.step + 1] = agent_perm.copy()
        self.topic_perm[self.step + 1] = topic_perm.copy()

        self.step = (self.step + 1) % self.episode_length


    def compute_returns(self, batch, next_value, value_normalizer=None):
        """
        報酬の割引和として、または GAE を使用してリターンを計算します。
        :param next_value: (np.ndarray) 最後のエピソードステップの次のステップの値予測。
        :param value_normalizer: (PopArt) Noneでない場合、PopArt値のノーマライザインスタンス。
        """

        self.value_preds[-1][batch][self.mask[-1][batch]] = next_value
        gae = 0

        for step in reversed(range(self.episode_length)):
            delta = self.rewards[step][batch][self.mask[step][batch]] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1][batch][self.mask[step + 1][batch]]) - value_normalizer.denormalize(self.value_preds[step][batch][self.mask[step][batch]])
                
            gae = delta + self.gamma * self.gae_lambda * gae

            self.advantages[step][batch][self.mask[step][batch]] = gae
            self.returns[step][batch][self.mask[step][batch]] = gae + value_normalizer.denormalize(self.value_preds[step][batch][self.mask[step][batch]])

    
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
        
        # keep (num_agent, dim)
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        mask = self.mask[:-1].reshape(-1, *self.mask.shape[2:])

        # obs.shape = (960, 90, 6564)
        # mask.shape = (960, 90)

        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        # actions.shape = (960, 90, 1)

        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        # value_preds.shape = (960, 90, 1)

        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        # returns.shape = (960, 90, 1)

        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        # action_log_probs.shape = (960, 90, 1)

        advantages = advantages.reshape(-1, *advantages.shape[2:])
        # advantages.shape = (960, 90, 1)

        #  一旦 データのシャッフルを停止
        #  複数 env での学習の際に効果があるのかを再検証
        """
        # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
        #  L: episode_length, T: n_rollout_threads, N: num_agent?
        obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
        mask_batch = mask[indices].reshape(-1, *mask.shape[1:])
        actions_batch = actions[indices].reshape(-1, *actions.shape[2:])

        # obs_batch.shape = (86400, 6564)
        # mask_batch.shape = (960, 90)
        # actions_batch.shape = (86400, 1)

        value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
        return_batch = returns[indices].reshape(-1, *returns.shape[2:])
        old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])

        #value_preds_batch.shape = (86400, 1)
        # returns_batch.shape = (86400, 1)
        # old_action_log_probs_batch = (86400, 1)

        if advantages is None:
            adv_targ = None
        else:
            adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])
            # adv_targ.shape = (86400, 1)

        return obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, mask_batch
        """

        # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
        #  L: episode_length, T: n_rollout_threads, N: num_agent?
        obs_batch = obs.reshape(-1, *obs.shape[2:])
        mask_batch = mask.reshape(-1, *mask.shape[1:])
        actions_batch = actions.reshape(-1, *actions.shape[2:])

        # obs_batch.shape = (86400, 6564)
        # mask_batch.shape = (960, 90)
        # actions_batch.shape = (86400, 1)

        value_preds_batch = value_preds.reshape(-1, *value_preds.shape[2:])
        return_batch = returns.reshape(-1, *returns.shape[2:])
        old_action_log_probs_batch = action_log_probs.reshape(-1, *action_log_probs.shape[2:])

        #value_preds_batch.shape = (86400, 1)
        # returns_batch.shape = (86400, 1)
        # old_action_log_probs_batch = (86400, 1)

        if advantages is None:
            adv_targ = None
        else:
            adv_targ = advantages.reshape(-1, *advantages.shape[2:])
            # adv_targ.shape = (86400, 1)

        return obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, mask_batch