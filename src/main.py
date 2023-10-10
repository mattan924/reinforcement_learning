import happo_trainer
import coma_trainer
from MAT.mat_runner import MATRunner
import sys


#learning_data_index_path = "../dataset/debug/debug/index/index_easy.csv"
learning_data_index_dir = "../dataset/dataset_hard/train/index/"
test_data_index_dir = "../dataset/dataset_hard/test/index/"

# 各種パラメーター
#  MAT
max_epi_itr = 10000
batch_size = 16
backup_itr = 100



device = "cuda:0"
result_dir = "../result/temporary/train_hard_dataset/"
file_name = "dataset10000_batch16_epoch4_"
output_base = result_dir + file_name
transformer_weight_base = "transformer"
#load_parameter_path = '../result/temporary/debug/hard/model_parameter/transformer_hard_mat_batch3_extend0_5000.pth'
load_parameter_path = None

start_epi_itr = 0

max_agent = 30
max_topic = 3

for i in range(1):
    output = output_base + str(i)
    transformer_weight = transformer_weight_base + "_" + file_name + str(i)

    #  標準エラー出力先の変更
    sys.stderr = open(output + "_err.log", 'w')

    #runner = MATRunner(max_epi_itr, batch_size, device, result_dir, backup_itr, max_agent, max_topic, learning_data_index_path=learning_data_index_path)
    #runner.train_single_env(output, transformer_weight, start_epi_itr, load_parameter_path=load_parameter_path)
    runner = MATRunner(max_epi_itr, batch_size, device, result_dir, backup_itr, max_agent, max_topic, learning_data_index_dir=learning_data_index_dir, test_data_index_dir=test_data_index_dir)
    runner.train_multi_env(output, transformer_weight, start_epi_itr, load_parameter_path=load_parameter_path)

#  標準エラー出力先を戻す
sys.stderr = sys.__stderr__
