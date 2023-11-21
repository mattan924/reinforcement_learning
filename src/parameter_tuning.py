from MAT.mat_runner import MATRunner
import torch
import optuna
import sys
import os
import logging


def objective(trial):
    # ハイパーパラメータ
    obs_size = 27
    batch_size = 15
    ppo_epoch = 6
    lr = trial.suggest_float('lr', 1e-06, 1e-02, log=True)
    eps = trial.suggest_float('eps', 1e-05, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    n_block = 6
    n_embd = 9
    reward_scaling = False

    os.makedirs(log_dir + "trial" + str(trial.number))

    runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd, reward_scaling, device, max_agent, max_topic)
    reward = runner.tuning_multi_env(trial, start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, log_dir, log_name_base, process_name)

    torch.cuda.empty_cache()

    return reward


#learning_data_index_path = "../dataset/debug/debug/index/index_easy.csv"
learning_data_index_dir = "../dataset/similar_dataset/easy/small15_select/train/index/"
test_data_index_dir = "../dataset/similar_dataset/easy/small15_select/test/index/"

# 設定
max_agent = 30
max_topic = 3
device = "cuda:1"
start_epi_itr = 0
max_epi_itr = 10000

log_dir = "../result/temporary/parameter_tuning/similar_dataset/small15_select/"
log_name_base = 'tuning_'
process_name = "process4"

optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(log_dir + log_name_base + process_name + '.log'))

pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1000)
study = optuna.create_study(storage="sqlite:///similar_dataset.db", study_name="small15_select", pruner=pruner, direction='minimize', load_if_exists=True)
study.optimize(objective, n_trials=10)
