from MAT.mat_runner import MATRunner
import optuna


def objective(trial):
    # ハイパーパラメータ
    obs_size = 27
    batch_size = 16
    ppo_epoch = 16
    lr = 0.0005
    eps = 1e-05
    weight_decay = 0
    n_block = 1
    n_embd = 9
    reward_scaling = False


    runner = MATRunner(obs_size, max_epi_itr, batch_size, ppo_epoch, device, result_dir, backup_itr, max_agent, max_topic, lr, eps, weight_decay, n_block, n_embd, learning_data_index_path=learning_data_index_path)
    runner.train_single_env(output, transformer_weight, start_epi_itr, reward_scaling, load_parameter_path=load_parameter_path)

    #runner = MATRunner(obs_size, max_epi_itr, batch_size, ppo_epoch, device, result_dir, backup_itr, max_agent, max_topic, lr, eps, weight_decay, n_block, n_embd, learning_data_index_dir=learning_data_index_dir, test_data_index_dir=test_data_index_dir)
    #runner.train_multi_env(output, transformer_weight, start_epi_itr, reward_scaling, load_parameter_path=load_parameter_path)

    return reward


learning_data_index_path = "../dataset/debug/debug/index/index_onetopic.csv.csv"
#learning_data_index_dir = "../dataset/debug/easy/regular_meeting/train/index/"
#test_data_index_dir = "../dataset/debug/easy/regular_meeting/test/index/"


device = "cuda:1"
max_epi_itr = 1000

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)