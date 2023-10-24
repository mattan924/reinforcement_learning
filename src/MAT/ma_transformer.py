import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from MAT.utils.util import check, init


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, n_topic, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head  #  1

        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))

        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent*n_topic + 1, n_agent*n_topic + 1)).view(1, 1, n_agent*n_topic + 1, n_agent*n_topic + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()
        #  B = 1
        #  L = num_agent*num_topic
        #  D = n_embd

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent, n_topic):
        super(EncodeBlock, self).__init__()

        self.attn = SelfAttention(n_embd, n_head, n_agent, n_topic, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = x + self.mlp(x)

        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent, n_topic):
        super(DecodeBlock, self).__init__()

        self.attn1 = SelfAttention(n_embd, n_head, n_agent, n_topic, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, n_topic, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = x + self.attn1(x, x, x)
        x = rep_enc + self.attn2(key=x, value=x, query=rep_enc)
        x = x + self.mlp(x)

        return x


class Encoder(nn.Module):

    def __init__(self, obs_distri_dim, obs_info_dim, n_block, n_embd, n_head, n_agent, n_topic, device):
        super(Encoder, self).__init__()

        self.obs_distri_dim = obs_distri_dim
        self.obs_info_dim = obs_info_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.n_topic = n_topic
        self.device = device

        self.obs_encoder_posi = nn.Sequential(init_(nn.Linear(obs_distri_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder_client = nn.Sequential(init_(nn.Linear(obs_distri_dim*3, n_embd), activate=True), nn.GELU())
        self.obs_encoder_edge = nn.Sequential(init_(nn.Linear(obs_distri_dim*5, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(init_(nn.Linear(n_embd*3+self.obs_info_dim, n_embd), activate=True), nn.GELU())

        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent, n_topic) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), init_(nn.Linear(n_embd, 1)))
        

    def forward(self, obs_posi, obs_client, obs_edge, obs_topic_info, mask):
        batch_dim = obs_posi.shape[0]
        
        obs_emb_posi = self.obs_encoder_posi(obs_posi)
        obs_emb_client = self.obs_encoder_client(obs_client)
        obs_emb_edge = self.obs_encoder_edge(obs_edge)

        obs_emb_posi = obs_emb_posi.unsqueeze(2).repeat(1, 1, self.n_topic, 1).reshape(batch_dim, self.n_agent*self.n_topic, self.n_embd)
        obs_emb_client = obs_emb_client.unsqueeze(1).repeat(1, self.n_agent, 1, 1).reshape(batch_dim, self.n_agent*self.n_topic, self.n_embd)
        obs_emb_edge = obs_emb_edge.unsqueeze(1).repeat(1, self.n_agent, 1, 1).reshape(batch_dim, self.n_agent*self.n_topic, self.n_embd)
        obs_topic_info = obs_topic_info.unsqueeze(1).repeat(1, self.n_agent, 1, 1).reshape(batch_dim, self.n_agent*self.n_topic, 3)

        obs_embeddings = self.obs_encoder(torch.cat([obs_emb_posi[mask], obs_emb_client[mask], obs_emb_edge[mask], obs_topic_info[mask]], dim=-1))

        rep = torch.zeros((batch_dim, self.n_agent*self.n_topic, self.n_embd), device=self.device)

        rep[mask] = self.blocks(obs_embeddings.reshape(batch_dim, -1, self.n_embd)).reshape(-1, self.n_embd)
        v_loc = self.head(rep[mask])

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, action_dim, n_block, n_embd, n_head, n_agent, n_topic):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd

        self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True), nn.GELU())
        
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent, n_topic) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), init_(nn.Linear(n_embd, action_dim)))


    #  state, action, and return
    def forward(self, action, obs_rep):
        batch_dim, action_len, _ = action.shape

        action_embeddings = self.action_encoder(action)
        #  action_embeddings.shape = torch.Size([1, num_agent*num_topic, n_embd])

        x = self.ln(action_embeddings)

        for block in self.blocks:
            #  x.shape = torch.Size([1, num_agent*num_topic, n_embd])
            #  obs_rep.shape = torch.Size([1, num_agent*num_topic, n_embd])
            x = block(x, obs_rep)

        #  x.shape = torch.Size([Batch, num_agent, n_embd=64])

        logit = self.head(x)
        #  logit.shape = torch.Size([Batch, num_agent, action_dim=14])

        return logit


class MultiAgentTransformer(nn.Module):

    def __init__(self, obs_distri_dim, obs_info_dim, action_dim, batch_size, max_agent, max_topic, n_block, n_embd, device=torch.device("cpu")):

        super(MultiAgentTransformer, self).__init__()

        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_agent = max_agent
        self.max_topic = max_topic
        #  dictionary の作成
        #  self.tpdv = {'dtype': torch.float32, 'device': device(type='cpu')}
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        self.n_block = n_block
        self.n_embd = n_embd
        self.n_head = 1

        self.encoder = Encoder(obs_distri_dim, obs_info_dim, self.n_block, self.n_embd, self.n_head, max_agent, max_topic, self.device)
        self.decoder = Decoder(action_dim, self.n_block, self.n_embd, self.n_head, max_agent, max_topic)

        self.to(device)


    def forward(self, obs_posi, obs_client, obs_edge, obs_topic_info, action, mask):
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        v_loc, obs_rep = self.encoder(obs_posi, obs_client, obs_edge, obs_topic_info, mask)

        action = action.long()

        action_log, entropy = self.discrete_parallel_act(obs_rep, action, mask)

        return action_log, v_loc, entropy


    def get_actions(self, obs_posi, obs_client, obs_edge, obs_topic_info, mask, deterministic=False):

        obs_posi = check(obs_posi).to(self.device)
        obs_client = check(obs_client).to(self.device)
        obs_edge = check(obs_edge).to(self.device)
        obs_topic_info = check(obs_topic_info).to(self.device)
        mask = check(mask).to(self.device)

        v_loc, obs_rep = self.encoder(obs_posi, obs_client, obs_edge, obs_topic_info, mask)
        
        output_action, output_action_log = self.discrete_autoregreesive_act(obs_rep, mask, deterministic=deterministic)

        return output_action, output_action_log, v_loc


    def get_values(self, obs_posi, obs_client, obs_edge, obs_topic_info, mask):

        obs_posi = check(obs_posi).to(**self.tpdv)
        obs_client = check(obs_client).to(**self.tpdv)
        obs_edge = check(obs_edge).to(**self.tpdv)
        obs_topic_info = check(obs_topic_info).to(**self.tpdv)
        mask = check(mask).to(self.device)

        v_tot, obs_rep = self.encoder(obs_posi, obs_client, obs_edge, obs_topic_info, mask)
        
        return v_tot
    

    def discrete_autoregreesive_act(self, obs_rep, mask, deterministic=False):
        batch_dim, max_action_len, _ = obs_rep.shape

        action_len = obs_rep[mask].reshape(batch_dim, -1, self.n_embd).shape[1]

        shifted_action = torch.zeros((batch_dim, action_len, self.action_dim + 1)).to(**self.tpdv)
        shifted_action[:, 0, 0] = 1

        output_action = torch.zeros((batch_dim, max_action_len, 1), dtype=torch.long)
        tmp_action = torch.zeros((batch_dim, action_len, 1), dtype=torch.long)

        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
        tmp_action_log = torch.zeros((batch_dim, action_len, 1), dtype=torch.float32)

        for i in range(action_len):
            #  decoder の出力から agent i のものを取り出す
            #  shifted_action 自分より前の agent の行動の onehot_vector が入っている

            #  shifted_action.shape = torch.Size([1, num_agent*num_topic, N_action+1])
            #  obs_rep.shape = torch.Size([1, num_agent*num_topic, n_embd])
            logit = self.decoder(shifted_action, obs_rep[mask].reshape(batch_dim, -1, self.n_embd))[:, i, :]

            distri = Categorical(logits=logit + 1e-8)

            if deterministic:
                action = distri.probs.argmax(dim=-1)
            else:
                action = distri.sample()

            action_log = distri.log_prob(action)
            #  action_log.shape = torch.Size([n_rollout_threads])

            tmp_action[:, i, :] = action.unsqueeze(-1)
            tmp_action_log[:, i, :] = action_log.unsqueeze(-1)

            #  action と action_log を格納

            if i + 1 < action_len:
                shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=self.action_dim)

        output_action[mask] = tmp_action.reshape(-1, 1)
        output_action_log[mask] = tmp_action_log.reshape(-1, 1)

        return output_action, output_action_log


    #  まとめて行動を選択
    def discrete_parallel_act(self, obs_rep, action, mask):
        #  mask.shape = (960, 15)
        #  action.shape = torch.Size([960, 15, 1])
        batch = obs_rep.shape[0]
        
        one_hot_action = F.one_hot(action[mask].reshape(batch, -1, 1).squeeze(-1), num_classes=self.action_dim).to(**self.tpdv)

        action_len = one_hot_action.shape[1]

        shifted_action = torch.zeros((batch, action_len, self.action_dim + 1)).to(**self.tpdv)
        shifted_action[:, 0, 0] = 1

        shifted_action[:, 1:, 1:] = one_hot_action[:, :-1]

        logit = self.decoder(shifted_action, obs_rep[mask].reshape(batch, -1, self.n_embd))

        distri = Categorical(logits=logit)
        action_log = distri.log_prob(action[mask].reshape(batch, -1)).unsqueeze(-1)
        entropy = distri.entropy().unsqueeze(-1)

        return action_log, entropy
    