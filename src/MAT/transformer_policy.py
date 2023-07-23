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

    def get_values(self, obs):
        """
        値関数の予測値を取得します。
        """

        obs = obs.reshape(-1, self.num_agents*self.num_topic, self.obs_dim)
        #  obs.shape = (1, num_agent*num_topic, obs_dim=2255)

        values = self.transformer.get_values(obs)

        values = values.view(-1, 1)
        #  values.shape = torch.Size([num_agent*num_topic, 1])

        return values
    

    def evaluate_actions(self, obs, actions):
        """
        アクタ更新のためのアクションログ確率 / エントロピー / value関数予測を取得します。
        :param obs (np.ndarray): アクタへのローカルエージェント入力．
        :param actions: (np.ndarray) 対数確率とエントロピーを計算するアクション。

        :return values: (torch.Tensor) 値関数の予測値。
        :return action_log_probs: (torch.Tensor) 入力アクションのログ確率。
        :return dist_entropy: (torch.Tensor) 与えられた入力に対するアクションの分布エントロピー。
        """

        obs = obs.reshape(-1, self.num_agents*self.num_topic, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents*self.num_topic, self.act_num)

        action_log_probs, values, entropy = self.transformer(obs, actions)

        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

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

