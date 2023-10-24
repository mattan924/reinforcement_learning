import happo_trainer
import coma_trainer
from MAT.mat_runner import MATRunner
import sys


#learning_data_index_path = "../dataset/debug/debug/index/index_easy_hight_load.csv"
learning_data_index_dir = "../dataset/debug/easy/regular_meeting/train/index/"
test_data_index_dir = "../dataset/debug/easy/regular_meeting/test/index/"

# 各種パラメーター
#  MAT
start_epi_itr = 0
max_epi_itr = 10
backup_itr = 100

max_agent = 30
max_topic = 3

# ハイパーパラメーター
obs_size = 27
batch_size = 16
ppo_epoch = 16
lr = 0.0005
eps = 1e-05
weight_decay = 0
n_block = 1
n_embd = 9
reward_scaling = False


device = "cuda:0"
result_dir = "../result/temporary/debug/"
file_name = "debug"
output_base = result_dir + file_name
transformer_weight_base = "transformer"
#load_parameter_path = '../result/temporary/debug/hard/model_parameter/transformer_hard_mat_batch3_extend0_5000.pth'
load_parameter_path = None


runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd, device, backup_itr, max_agent, max_topic)

for i in range(1):
    output = output_base + str(i)
    transformer_weight = transformer_weight_base + "_" + file_name + str(i)

    #  標準エラー出力先の変更
    #sys.stderr = open(output + "_err.log", 'w')

    #runner.train_single_env(start_epi_itr, max_epi_itr, learning_data_index_path, result_dir, output, transformer_weight, reward_scaling, load_parameter_path=load_parameter_path)
    runner.train_multi_env(start_epi_itr, max_epi_itr, learning_data_index_dir, test_data_index_dir, result_dir, output, transformer_weight, reward_scaling, load_parameter_path=load_parameter_path)

#  標準エラー出力先を戻す
sys.stderr = sys.__stderr__
