import torch
import numpy as np
import torch.nn.functional as F


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

        self.value_preds = np.zeros((self.episode_length + 1, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros((self.episode_length, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)
        
        self.actions = np.zeros((self.episode_length, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.batch_size, num_agents*num_topic, 1), dtype=np.float32)

        self.agent_perm = np.zeros((self.episode_length + 1, self.batch_size, self.num_agents))
        self.topic_perm = np.zeros((self.episode_length + 1, self.batch_size, self.num_topic))

        self.step = 0

    def insert(self, batch, obs, actions, action_log_probs, value_preds, rewards):
        """
        バッファにデータを挿入します。
        param obs: (np.ndarray) ローカルエージェントのオブザベーション。
        :param actions:(np.ndarray) エージェントが取った行動。
        param action_log_probs:(np.ndarray) エージェントが取った行動のログ確率。
        param value_preds: (np.ndarray) 各ステップにおける値関数の予測値．
        param rewards: (np.ndarray) 各ステップで収集した報酬。
        """

        self.obs[self.step + 1][batch] = obs.reshape(self.num_agents*self.num_topic, self.obs_dim).copy()
        self.actions[self.step][batch] = actions.reshape(self.num_agents*self.num_topic, 1).copy()
        self.action_log_probs[self.step][batch] = action_log_probs.copy()
        self.value_preds[self.step][batch] = value_preds.copy()
        self.rewards[self.step][batch] = rewards.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length


    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()


    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()


    def compute_returns(self, batch, next_value, value_normalizer=None):
        """
        報酬の割引和として、または GAE を使用してリターンを計算します。
        :param next_value: (np.ndarray) 最後のエピソードステップの次のステップの値予測。
        :param value_normalizer: (PopArt) Noneでない場合、PopArt値のノーマライザインスタンス。
        """

        self.value_preds[-1][batch] = next_value
        gae = 0

        for step in reversed(range(self.episode_length)):
            delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) - value_normalizer.denormalize(self.value_preds[step])
                
            gae = delta + self.gamma * self.gae_lambda * gae

            self.advantages[step] = gae
            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])


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
            
            mini_batch_size = self.batch_size // num_mini_batch

        rand = torch.randperm(self.batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        
        rows, cols = _shuffle_agent_grid(self.batch_size, self.num_agents*self.num_topic)

        # keep (num_agent, dim)
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]

        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            #  L: episode_length, T: n_rollout_threads, N: num_agent?
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])

            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            #  yield 文: return 文みたいに値を返すが、関数は終了せず for 文を継続する
            yield obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ
