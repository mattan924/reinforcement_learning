from MAT.mat_runner import MATRunner
import torch
import optuna
import sys
import logging


def objective(trial):
    # ハイパーパラメータ
    obs_size = 27
    batch_size = 16
    ppo_epoch = 8
    lr = trial.suggest_float('lr', 1e-07, 1e-02, log=True)
    eps = trial.suggest_float('eps', 1e-06, 1e1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1, log=True)
    n_block = 1
    n_embd = 9
    reward_scaling = False

    runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd, reward_scaling, device, max_agent, max_topic)
    reward = runner.tuning_single_env(trial, start_epi_itr, max_epi_itr, learning_data_index_path, log_name_base, process_name)

    torch.cuda.empty_cache()

    return reward


learning_data_index_path = "../dataset/debug/debug/index/index_easy.csv"
#learning_data_index_dir = "../dataset/debug/easy/regular_meeting/train/index/"
#test_data_index_dir = "../dataset/debug/easy/regular_meeting/test/index/"

# 設定
max_agent = 30
max_topic = 3
device = "cuda:1"
start_epi_itr = 0
max_epi_itr = 2000

log_name_base = '../result/temporary/parameter_tuning/easy_tuning_importance_'
process_name = "process4"

optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(log_name_base + process_name + '.log'))

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
study = optuna.create_study(storage="sqlite:///easy_tuning.db", study_name="easy_parallel_importance", pruner=pruner, direction='minimize', load_if_exists=True)
study.optimize(objective, n_trials=20)
