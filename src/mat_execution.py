from env import Env
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
max_epi_itr = 4400
backup_itr = 100

max_agent = 40
max_topic = 3

# ハイパーパラメーター
obs_size = 9
batch_size = 1
ppo_epoch = 6
lr = 0.0005
eps = 1e-05
weight_decay = 0
n_block = 3
n_embd1 = 81
n_embd2 = 9
reward_scaling = True


device = "cuda:1"
data_index_path = "../dataset/master_thesis/multi_data/high_capacity_low_cycle_client20_fix20/test/index/data_fix_traking1_assign8_edge0_topic2.csv"
load_parameter_path_base = "../result/save/master_thesis/multi_data/high_capacity_low_cycle/model_parameter/transformer_client20_fix20_0_"

runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd1, n_embd2, reward_scaling, device, max_agent, max_topic)

for epi in range(start_epi_itr, max_epi_itr+1, backup_itr):
    print(f"========== epi: {epi} ==========")
    load_parameter_path = load_parameter_path_base + str(epi) + ".pth"
    
    reward = runner.execute_single_env(data_index_path, load_parameter_path)

    #print(f"{reward}:{epi}/{max_epi_itr} is complete")

#animation.create_single_assign_animation(data_index_path, output_animation_file, FPS=5)
