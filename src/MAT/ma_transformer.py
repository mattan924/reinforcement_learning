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

    def __init__(self, n_embd, n_head, n_agent, masked=False):
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
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()
        #  B = 100 or 1
        #  L = 6 (agent number ?)
        #  D = 64 (埋め込み後の次元 ?)

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

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim  #  172
        self.n_embd = n_embd  #  64
        self.n_agent = n_agent  #  6
        #  n_block = 1
        #  n_head = 1
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim), init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))
        

    def forward(self, obs):
        # obs: (batch, n_agent, obs_dim)

        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd


        self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True), nn.GELU())
        
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim), init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))


    # state, action, and return
    def forward(self, action, obs_rep):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        #  action = torch.Size([Batch, num_agent, action_dim + 1])

        action_embeddings = self.action_encoder(action)

        #  action_embeddings.shape = torch.Size([Batch, num_agent, n_embd=64])
        x = self.ln(action_embeddings)
        #  x.shape = torch.Size([Batch, num_aget, n_embd=64])

        for block in self.blocks:
            x = block(x, obs_rep)

        #  x.shape = torch.Size([Batch, num_agent, n_embd=64])

        logit = self.head(x)
        #  logit.shape = torch.Size([Batch, num_agent, action_dim=14])

        return logit


class MultiAgentTransformer(nn.Module):

    def __init__(self, obs_dim, action_dim, batch_size, n_agent, n_topic, device=torch.device("cpu")):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.n_topic = n_topic
        self.action_dim = action_dim
        self.batch_size = batch_size
        #  dictionary の作成
        #  self.tpdv = {'dtype': torch.float32, 'device': device(type='cpu')}
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        self.n_block = 1
        self.n_embd = 128
        self.n_head = 1

        self.encoder = Encoder(obs_dim, self.n_block, self.n_embd, self.n_head, n_agent)
        self.decoder = Decoder(obs_dim, action_dim, self.n_block, self.n_embd, self.n_head, n_agent)

        self.to(device)


    def forward(self, obs, action, available_actions=None):
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        v_loc, obs_rep = self.encoder(obs)

        action = action.long()
        action_log, entropy = self.discrete_parallel_act(self.decoder, obs_rep, action, self.batch_size, self.n_agent, self.action_dim, self.tpdv, available_actions)

        return action_log, v_loc, entropy


    def get_actions(self, obs, deterministic=False):
        #  torch.float32, cpu へ変換
        obs = check(obs).to(**self.tpdv)

        print(f"obs.shape = {obs.shape}")

        #  obs を Encoder を用いてエンコード
        #  obs.shape = torch.Size([1, num_agents*num_topic, obs_dim=2255])
        v_loc, obs_rep = self.encoder(obs)
        #  v_loc.shape = torch.Size([1, num_agents*num_topic, 1])
        #  obs_rep = torch.Size([1, num_agents*num_topic, n_embd])
        
        output_action, output_action_log = self.discrete_autoregreesive_act(self.decoder, obs_rep, deterministic=deterministic)

        return output_action, output_action_log, v_loc


    def get_values(self, state, obs):

        obs = check(obs).to(**self.tpdv)

        v_tot, obs_rep = self.encoder(state, obs)
        
        return v_tot
    

    def discrete_autoregreesive_act(self, decoder, obs_rep, deterministic=False):
        batch_dim = obs_rep.shape[0]

        shifted_action = torch.zeros((batch_dim, self.n_agent*self.n_topic, self.action_dim + 1)).to(**self.tpdv)
        shifted_action[:, 0, 0] = 1

        output_action = torch.zeros((batch_dim, self.n_agent*self.n_topic, 1), dtype=torch.long)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

        for i in range(self.n_agent*self.n_topic):
            #  decoder の出力から agent i のものを取り出す
            #  shifted_action 自分より前の agent の行動の onehot_vector が入っている
           
            #  shifted_action.shape = torch.Size([1, num_agent*num_topic, N_action+1])
            #  obs_rep.shape = torch.Size([1, num_agent*num_topic, n_embd])
            logit = decoder(shifted_action, obs_rep)[:, i, :]

            distri = Categorical(logits=logit)

            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            #  action.shape = torch.Size([n_rollout_threads])

            action_log = distri.log_prob(action)
            #  action_log.shape = torch.Size([n_rollout_threads])

            output_action[:, i, :] = action.unsqueeze(-1)
            output_action_log[:, i, :] = action_log.unsqueeze(-1)
            #  action と action_log を格納

            if i + 1 < self.n_agent*self.n_topic:
                shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=self.action_dim)

        return output_action, output_action_log


    #  まとめて行動を選択
    def discrete_parallel_act(self, decoder, obs_rep, action, batch_size, n_agent, action_dim, tpdv, available_actions=None):
        
        #  action_dim = 14
        #  action.shape = torch.Size([rollout_threads*episode_length, num_agent])
        one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
        #  one_hot_action.shape = torch.Size([rollout_threads*episode_length, num_agent, action_dim])

        shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
        shifted_action[:, 0, 0] = 1
        shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
        logit = decoder(shifted_action, obs_rep)

        #  available_actions is not None
        if available_actions is not None:
            logit[available_actions == 0] = -1e10

        distri = Categorical(logits=logit)
        action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = distri.entropy().unsqueeze(-1)

        return action_log, entropy
    