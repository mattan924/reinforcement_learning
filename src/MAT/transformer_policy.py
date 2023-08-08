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

    def __init__(self, obs_dim, obs_distri_dim, obs_info_dim, act_dim, batch_size, num_agents, num_topic, max_agent, max_topic, device=torch.device("cpu")):
        self.device = device
        self.lr = 0.0005
        self.opti_eps = 1e-05
        self.weight_decay = 0
        self._use_policy_active_masks = True

        self.obs_dim = obs_dim
        self.obs_distri_dim = obs_distri_dim
        self.obs_info_dim = obs_info_dim

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
        self.transformer = MAT(self.obs_distri_dim, self.obs_info_dim, self.act_dim, self.batch_size, self.num_agents, self.num_topic, max_agent, max_topic, device=device)


        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)


    def get_actions(self, obs, mask, near_action, deterministic=False):
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

        obs = obs[:, mask]

        actions, action_log_probs, action_distribution, values = self.transformer.get_actions(obs, near_action, mask, deterministic)
        
        actions = actions.view(-1, self.act_num)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        return values, actions, action_log_probs, action_distribution

    def get_values(self, obs, mask):
        """
        値関数の予測値を取得します。
        """

        obs = obs.reshape(-1, self.num_agents*self.num_topic, self.obs_dim)
        #  obs.shape = (1, num_agent*num_topic, obs_dim)

        obs = obs[:, mask]

        values = self.transformer.get_values(obs)

        values = values.view(-1, 1)
        #  values.shape = torch.Size([num_agent*num_topic, 1])

        return values
    

    def evaluate_actions(self, obs, actions, mask):
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

        action_log_probs, values, entropy = self.transformer(obs, actions, mask)

        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

        entropy = entropy.mean()

        return values, action_log_probs, entropy


    def save(self, save_dir, transformer_weight, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/" + transformer_weight + "_" + str(episode) + ".pth")


    def restore(self, model_path):
        print(f"model_dir = {model_path}")
        self.transformer.load_state_dict(torch.load(model_path))


    def train(self):
        self.transformer.train()
        

    def eval(self):
        self.transformer.eval()

