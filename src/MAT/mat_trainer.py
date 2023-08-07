import numpy as np
import torch
import torch.nn as nn
from MAT.utils.util import get_gard_norm, huber_loss, check
from MAT.utils.valuenorm import ValueNorm


class MATTrainer:
    """
    MAT がポリシーを更新するためのトレーナークラスです。
    param args: (argparse.Namespace) 関連するモデル、ポリシー、env 情報を含む引数です。
    param policy: (R_MAPPO_Policy) 更新するポリシーを指定します。
    param device: (torch.device) 実行するデバイスを指定します (cpu/gpu)。
    """
    def __init__(self, policy, num_agents, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy  #  transformer_policy の TransformerPolicy
        self.num_agents = num_agents

        self.clip_param = 0.05
        self.ppo_epoch = 16
        self.num_mini_batch = 1
        self.data_chunk_length = 10
        self.value_loss_coef = 1
        self.entropy_coef = 0.01
        self.max_grad_norm = 10.0    
        self.huber_delta = 10.0

        self._use_max_grad_norm = True
        self._use_clipped_value_loss = True
        self._use_huber_loss = True
        self._use_valuenorm = True
        self._use_value_active_masks = False
        self._use_policy_active_masks = True

        self.value_normalizer = ValueNorm(1, device=self.device)


    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        value 関数の損失を計算します。
        :param values: (torch.Tensor) value 関数の予測値。
        :param value_preds_batch: (torch.Tensor) バッチデータからの "古い"予測値 (value クリップ損失に利用).
        :param return_batch: (torch.Tensor) return to go returns
        :param active_masks_batch: (torch.Tensor) 与えられたタイムステップでエージェントがアクティブかデッドかを表す。

        :return value_loss: (torch.Tensor) value 関数の損失。
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        self.value_normalizer.update(return_batch)
        error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = self.value_normalizer.normalize(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss = value_loss.mean()

        return value_loss
    

    def ppo_update(self, sample):
        """
        アクターとクリティックのネットワークを更新します。
        param sample: (タプル) ネットワークを更新するデータバッチを含みます。
        :update_actor: (bool) アクターネットワークを更新するかどうか。

        :return value_loss: (torch.Tensor) 値関数の損失。
        :return critic_grad_norm: (torch.Tensor) 批評家up9dateからの勾配ノルム。
        :return policy_loss: (torch.Tensor) actor(policy)の損失値.
        :return dist_entropy: (torch.Tensor) アクタのエントロピー.
        :return actor_grad_norm: (torch.Tensor) アクタの更新からの勾配ノルム。
        :return imp_weights: (torch.Tensor) 重要度サンプリングの重み。
        """

        obs_batch, actions_batch, value_preds_batch, return_batch,old_action_log_probs_batch, adv_targ, mask_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(obs_batch, actions_batch, mask_batch)

        mask_batch = mask_batch.reshape(-1)
        
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch[mask_batch])

        surr1 = imp_weights * adv_targ[mask_batch]
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ[mask_batch]

        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch[mask_batch], return_batch[mask_batch])

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)

        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        ミニバッチGDを使用してトレーニング更新を実行します。
        :param buffer: (SharedReplayBuffer) トレーニングデータを含むバッファ。
        param update_actor: (bool) アクタネットワークを更新するかどうか。

        :return train_info: (dict)トレーニングの更新に関する情報（損失や勾配ノルムなど）を格納します。
        """

        advantages_copy = buffer.advantages.copy()
        mask = buffer.mask[:-1].copy()
        
        #  np.nanmean, np.nanstd: nan を含む配列の平均、標準偏差を求める
        mean_advantages = np.nanmean(advantages_copy[mask])
        std_advantages = np.nanstd(advantages_copy[mask])
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
        
        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
