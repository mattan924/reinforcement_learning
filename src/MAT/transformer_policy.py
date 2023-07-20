import torch
import numpy as np
from MAT.ma_transformer import MultiAgentTransformer as MAT
from MAT.utils.util import check
from MAT.ma_transformer import MultiAgentTransformer


class TransformerPolicy:
    """
    MAT Policyクラス。アクタとクリティックのネットワークをラップして、アクションと価値関数の予測を計算します。

    param args: (argparse.Namespace) モデルとポリシーの関連情報を含む引数．
    param obs_space: (gym.Space) 観測空間．
    param cent_obs_space: (gym.Space) 値関数入力空間（MAPPO は集中入力, IPPO は分散入力）．
    param action_space: (gym.Space) アクション空間．
    param device: (torch.device) 実行するデバイスを指定します（cpu/gpu）。
    """

    def __init__(self, obs_dim, act_dim, batch_size, num_agents, num_topic, device=torch.device("cpu")):
        self.device = device
        self.lr = 0.0005
        self.opti_eps = 1e-05
        self.weight_decay = 0
        self._use_policy_active_masks = True

        self.obs_dim = obs_dim

        self.act_dim = act_dim
        self.act_num = 1

        self.batch_size = batch_size
        
        #  obs_dim:  172
        #  share_obs_dim:  213
        #  act_dim:  14

        self.num_agents = num_agents
        self.num_topic = num_topic
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        #  MAT インスタンスの生成
        self.transformer = MAT(self.obs_dim, self.act_dim, self.batch_size, self.num_agents, self.num_topic,device=device)


        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)


    def get_actions(self, obs, deterministic=False):
        """
        与えられた入力に対するアクションと値関数の予測を計算します。
        param obs (np.ndarray): actor へのローカルエージェント入力．
        param deterministic (bool): アクションを分布のモードにするか、サンプリングするか。

        :return values: (torch.Tensor) 値関数の予測値。
        :return actions: (torch.Tensor) 取るべきアクション。
        :return action_log_probs: (torch.Tensor) 選択されたアクションのログ確率。
        """

        #  obs.shape = (num_agents, num_topic, obs_dim=2255)
        obs = obs.reshape(-1, self.num_agents*self.num_topic, self.obs_dim)
        #  obs.shape = (1, num_agent*num_topic, obs_dim=2255)

        actions, action_log_probs, values = self.transformer.get_actions(obs, deterministic)
        
        #  actions.shape = torch.Size([n_rollout_threads, num_agent, 1])
        #  actions_log_probs = torch.Size([n_rollout_threads, num_agent, 1])
        #  values.shape = torch.Size([n_rollout_threads, num_agent, 1])

        actions = actions.view(-1, self.act_num)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        #  actions.shape = torch.Size([n_rollout_threads*num_agent, 1])
        #  actions_log_probs = torch.Size([n_rollout_threads*num_agent, 1])
        #  values.shape = torch.Size([n_rollout_threads*num_agent, 1])

        return values, actions, action_log_probs

    def get_values(self, cent_obs, obs, rnn_states_critic, masks, available_actions=None):
        """
        値関数の予測値を取得します。
        :param cent_obs (np.ndarray): 評論家への集中入力。
        :param rnn_states_critic: (np.ndarray)criticがRNNの場合、criticのRNN状態。
        :param masks: (np.ndarray)RNNの状態をリセットするポイントを示す。

        :return values: (torch.Tensor)予測値の関数。
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        # cent_obs.shape = (n_rollout_threads, num_agent, share_obs_dim=213)
        # obs.shape = (n_rollout_threads, num_agent, obs_dim=172)
        # available_actions.shape = (n_rollout_threads, num_agent, action_dim=14)
        values = self.transformer.get_values(cent_obs, obs, available_actions)
        # values.shape = torch.Size([n_rollout_threads, num_agent, 1])

        values = values.view(-1, 1)
        # values.shape = torch.Size([n_rollout_threads*num_agent, 1])

        return values
    

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks, available_actions=None, active_masks=None):
        """
        アクタ更新のためのアクションログ確率 / エントロピー / value関数予測を取得します。
        :param cent_obs (np.ndarray): 評論家への集中入力．
        :param obs (np.ndarray): アクタへのローカルエージェント入力．
        :param rnn_states_actor: (np.ndarray) アクターがRNNの場合、アクターのRNN状態。
        :param rnn_states_critic: (np.ndarray)criticがRNNの場合、criticのRNN状態。
        :param actions: (np.ndarray) 対数確率とエントロピーを計算するアクション。
        :param masks: (np.ndarray)RNNの状態をリセットするポイントを示す。
        :param available_actions: (np.ndarray)エージェントが利用可能なアクションを示す。 (Noneの場合、全てのアクションが利用可能)
        :param active_masks: (torch.Tensor)エージェントがアクティブかデッドかを示す。

        :return values: (torch.Tensor) 値関数の予測値。
        :return action_log_probs: (torch.Tensor) 入力アクションのログ確率。
        :return dist_entropy: (torch.Tensor) 与えられた入力に対するアクションの分布エントロピー。
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents, self.act_num)

        #  cent_obs.shape = (episode_length*n_rollout_threads, num_agent, share_obs_dim=213)
        #  obs.shape = (episode_length*n_rollout_threads, num_agent, obs_dim=172)
        #  actions.shape = (episode_length*n_rollout_threads, num_agent, 1)

        #  available_action is not None
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        action_log_probs, values, entropy = self.transformer(cent_obs, obs, actions, available_actions)

        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

        #  action_log_probs.shape = torch.Size([episode_length*n_rollout_threads*num_agent, 1])
        #  values.shape = torch.Size([episode_length*n_rollout_threads*num_agent, 1])
        #  entropy.shape = torch.Size([episode_length*n_rollout_threads*num_agent, 1])

        #  self._use_policy_active_masks is True
        #  active_masks is not None
        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        return values, action_log_probs, entropy

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=True):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        # this function is just a wrapper for compatibility
        rnn_states_critic = np.zeros_like(rnn_states_actor)
        _, actions, _, rnn_states_actor, _ = self.get_actions(cent_obs,
                                                              obs,
                                                              rnn_states_actor,
                                                              rnn_states_critic,
                                                              masks,
                                                              available_actions,
                                                              deterministic)

        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

