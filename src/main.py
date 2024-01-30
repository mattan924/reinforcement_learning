import happo_trainer
import coma_trainer
from MAT.mat_runner import MATRunner
import sys


#learning_data_index_path = "../dataset/master_thesis/single_data/index/fix_allocation_low_capacity_high_cycle.csv"
learning_data_index_dir = "../dataset/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle_client20_fix20_data10000/train/index/"
test_data_index_dir = "../dataset/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle_client20_fix20_data10000/test/index/"

# 各種パラメーター
# MAT
start_epi_itr = 4400
max_epi_itr = 15000
backup_itr = 100

max_agent = 40
max_topic = 3

# ハイパーパラメーター
obs_size = 9
sample_data = 64
multi_env = 1
batch_size = sample_data * multi_env
ppo_epoch = 6
lr = 0.0005
eps = 1e-05
weight_decay = 0.0001
n_block = 3
n_embd1 = 81
n_embd2 = 9
reward_scaling = False

num_mini_batch = 4


device = "cuda:1"
result_dir = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/"
file_name = "client20_fix20_"
output_base = result_dir + file_name
transformer_weight_base = "transformer"
load_parameter_path = '../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/model_parameter/transformer_client20_fix20_0_4400.pth'
#load_parameter_path = None


runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd1, n_embd2, reward_scaling, num_mini_batch, device, max_agent, max_topic)

for i in range(1):
    output = output_base + str(i)
    transformer_weight = transformer_weight_base + "_" + file_name + str(i)

    #  標準エラー出力先の変更
    sys.stderr = open(output + "_err.log", 'w')

    #runner.train_single_env(start_epi_itr, max_epi_itr, learning_data_index_path, result_dir, output, transformer_weight, backup_itr, load_parameter_path=load_parameter_path)
    runner.train_multi_env(sample_data, start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, result_dir, output, transformer_weight, backup_itr, load_parameter_path=load_parameter_path)
    #runner.debug_multi_env(sample_data, start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, result_dir, output, transformer_weight, backup_itr, load_parameter_path=load_parameter_path)

#  標準エラー出力先を戻す
sys.stderr = sys.__stderr__
