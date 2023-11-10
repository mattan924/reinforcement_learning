from env import Env
from MAT.ma_transformer import MultiAgentTransformer
from MAT.transformer_policy import TransformerPolicy
from MAT.mat_trainer import MATTrainer
from MAT.shared_buffer import SharedReplayBuffer
from MAT.mat_runner import MATRunner

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import random
import torch
import numpy as np

sys.path.append("../../dataset_visualization/src")
import util
import animation
from data import DataSolution

 
# 各種パラメーター
# MAT

start_epi_itr = 0
max_epi_itr = 37000
backup_itr = 100

max_agent = 30
max_topic = 3

# ハイパーパラメーター
obs_size = 27
batch_size = 16
ppo_epoch = 6
lr = 0.0005
eps = 1e-05
weight_decay = 0
n_block = 1
n_embd = 9
reward_scaling = True


device = "cuda:1"
data_index_path = "../dataset/similar_dataset/easy/traking_assign/train/index/hight_load_traking1_assign1_edge0_topic0.csv"
load_parameter_path_base = "../result/temporary/similar_dataset/easy/traking_assign/model_parameter/transformer_hight_load_multi_scaling0_"
output_file = "../result/temporary/similar_dataset/easy/traking_assign/hight_load_multi_scaling0_traindata_execution2.log"

runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd, reward_scaling, device, max_agent, max_topic)

with open(output_file, "w") as f:
    pass

for epi in range(start_epi_itr, max_epi_itr+1, backup_itr):
    load_parameter_path = load_parameter_path_base + str(epi) + ".pth"
    
    reward = runner.execute_single_env(data_index_path, load_parameter_path)

    with open(output_file, "a") as f:
        f.write(f"{epi}, {reward}\n")

    #print(f"{reward}:{epi}/{max_epi_itr} is complete")

#animation.create_single_assign_animation(data_index_path, output_animation_file, FPS=5)
