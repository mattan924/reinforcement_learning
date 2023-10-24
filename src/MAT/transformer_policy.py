import torch
import numpy as np
from MAT.ma_transformer import MultiAgentTransformer as MAT
from MAT.ma_transformer_multi import MultiAgentTransformer as MAT_multi
from MAT.utils.util import check
from MAT.ma_transformer import MultiAgentTransformer
import time


class TransformerPolicy:
    """
    MAT Policyクラス。アクタとクリティックのネットワークをラップして、アクションと価値関数の予測を計算します。

    param args: (argparse.Namespace) モデルとポリシーの関連情報を含む引数．
    param obs_space: (gym.Space) 観測空間．
    param cent_obs_space: (gym.Space) 値関数入力空間（MAPPO は集中入力, IPPO は分散入力）．
    param action_space: (gym.Space) アクション空間．
    param device: (torch.device) 実行するデバイスを指定します（cpu/gpu）。
    """

    def __init__(self, obs_dim, obs_distri_dim, obs_info_dim, act_dim, batch_size, max_agent, max_topic, lr, eps, weight_decay, n_block, n_embd, device=torch.device("cpu"), multi=True):
        self.device = device
        self.lr = lr
        self.opti_eps = eps
        self.weight_decay = weight_decay
        self._use_policy_active_masks = True

        self.obs_dim = obs_dim
        self.obs_distri_dim = obs_distri_dim
        self.obs_info_dim = obs_info_dim

        self.act_dim = act_dim
        self.act_num = 1

        self.batch_size = batch_size

        self.max_agent = max_agent
        self.max_topic = max_topic
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        #  MAT インスタンスの生成
        if multi:
            self.transformer = MAT_multi(self.obs_distri_dim, self.obs_info_dim, self.act_dim, self.batch_size, max_agent, max_topic, n_block, n_embd, device=device)
        else:
            self.transformer = MAT(self.obs_distri_dim, self.obs_info_dim, self.act_dim, self.batch_size, max_agent, max_topic, n_block, n_embd, device=device)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)


    def get_actions(self, obs_posi, obs_client, obs_edge, obs_topic_info, mask, deterministic=False):
        """
        与えられた入力に対するアクションと値関数の予測を計算します。
        """
        batch, _, distri_dim = obs_posi.shape

        actions, action_log_probs, values = self.transformer.get_actions(obs_posi, obs_client, obs_edge, obs_topic_info, mask, deterministic)
        
        actions = actions.view(batch, -1, self.act_num)
        action_log_probs = action_log_probs.view(batch, -1, self.act_num)

        return values, actions, action_log_probs
    

    def get_values(self, obs_posi, obs_client, obs_edge, obs_topic_info, mask):
        """
        値関数の予測値を取得します。
        """

        values = self.transformer.get_values(obs_posi, obs_client, obs_edge, obs_topic_info, mask)

        return values
    

    def evaluate_actions(self, obs_posi, obs_client, obs_edge, obs_topic_info, actions, mask):
        """
        アクタ更新のためのアクションログ確率 / エントロピー / value関数予測を取得します。
        """

        obs_posi = obs_posi.reshape(-1, self.max_agent, self.obs_distri_dim)
        obs_client = obs_client.reshape(-1, self.max_topic, self.obs_distri_dim*3)
        obs_edge = obs_edge.reshape(-1, self.max_topic, self.obs_distri_dim*5)
        obs_topic_info = obs_topic_info.reshape(-1, self.max_topic, 3)
        actions = actions.reshape(-1, self.max_agent*self.max_topic, self.act_num)

        action_log_probs, values, entropy = self.transformer(obs_posi, obs_client, obs_edge, obs_topic_info, actions, mask)

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

