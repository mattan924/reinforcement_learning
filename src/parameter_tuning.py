from MAT.mat_runner import MATRunner
import torch
import optuna
import sys
import os
import logging


def objective(trial):
    # ハイパーパラメータ
    obs_size = 27
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    ppo_epoch = trial.suggest_categorical('ppo_epoch', [4 ,8, 16])
    lr = trial.suggest_float('lr', 1e-06, 1e-02, log=True)
    eps = trial.suggest_float('eps', 1e-06, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    n_block = trial.suggest_int('n_block', 1, 6)
    n_embd = 9
    reward_scaling = True

    os.makedirs(log_dir + "trial" + str(trial.number))

    runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd, reward_scaling, device, max_agent, max_topic)
    reward = runner.tuning_multi_env(trial, start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, log_dir, log_name_base, process_name)

    torch.cuda.empty_cache()

    return reward


#learning_data_index_path = "../dataset/debug/debug/index/index_easy.csv"
learning_data_index_dir = "../dataset/similar_dataset/easy/traking_assign/train/index/"
test_data_index_dir = "../dataset/similar_dataset/easy/traking_assign/test/index/"

# 設定
max_agent = 30
max_topic = 3
device = "cuda:0"
start_epi_itr = 0
max_epi_itr = 2000

log_dir = "../result/temporary/parameter_tuning/easy_hight_load_multi/traking_assign/"
log_name_base = 'tuning_'
process_name = "process8"

optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(log_dir + log_name_base + process_name + '.log'))

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
study = optuna.create_study(storage="sqlite:///easy_hight_load_multi_tuning.db", study_name="tuning", pruner=pruner, direction='minimize', load_if_exists=True)
study.optimize(objective, n_trials=5)
