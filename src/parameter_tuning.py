from MAT.mat_runner import MATRunner
import torch
import optuna
import sys
import os
import logging


def objective(trial):
    # ハイパーパラメータ
    obs_size =  9
    batch_size = 64
    ppo_epoch = 6
    lr = trial.suggest_float('lr', 1e-06, 1e-02, log=True)
    eps = trial.suggest_float('eps', 1e-05, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    n_block = 3
    n_embd1 = 81
    n_embd2 = 9
    reward_scaling = False

    os.makedirs(log_dir + "trial" + str(trial.number))

    runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd1, n_embd2, reward_scaling, device, max_agent, max_topic)
    reward = runner.tuning_multi_env(trial, start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, log_dir, log_name_base, process_name)

    torch.cuda.empty_cache()

    return reward


#learning_data_index_path = "../dataset/debug/debug/index/index_easy.csv"
learning_data_index_dir = "../dataset/master_thesis/multi_data/high_capacity_low_cycle_client20_fix20/train/index/"
test_data_index_dir = "../dataset/master_thesis/multi_data/high_capacity_low_cycle_client20_fix20/test/index/"

# 設定
max_agent = 20
max_topic = 3
device = "cuda:0"
start_epi_itr = 0
max_epi_itr = 1000

log_dir = "../result/save/master_thesis/multi_data/parameter_tuning/high_capacity_low_cycle/"
log_name_base = 'client20_fix20_tuning_'
process_name = "process1"

optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(log_dir + log_name_base + process_name + '.log'))

pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=25)
study = optuna.create_study(storage="sqlite:///high_low.db", study_name="client20_fix20", pruner=pruner, direction='minimize', load_if_exists=True)
study.optimize(objective, n_trials=10)
