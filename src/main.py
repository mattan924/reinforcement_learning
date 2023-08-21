import happo_trainer
import coma_trainer
from MAT.mat_runner import MATRunner
import sys


learning_data_index_path = "../dataset/debug/debug/index/index_hard.csv"
#learning_data_index_dir = "../dataset/debug/onetopic/train/index/"
#test_data_index_dir = "../dataset/debug/onetopic/test/index/"

# 各種パラメーター
#  HAPPO
# max_epi_itr = 3000
# buffer_size = 15000
# batch_size = 300
# eps_clip = 0.2
# backup_iter = 100

#  COMA
#max_epi_itr = 5000
#buffer_size = 3000
#batch_size = 300
#backup_iter = 100

#  MAT
max_epi_itr = 3000
batch_size = 16
backup_itr = 1000



device = "cuda:1"
result_dir = "../result/temporary/debug/hard/"
file_name = "hard_mat_batch"
output_base = result_dir + file_name
# actor_weight_base = "actor_weight"
# critic_weight_base = "critic_weight"
# V_net_weight_base = "V_net_weight"
transformer_weight_base = "transformer"
#load_parameter_path = '../result/temporary/debug/model_parameter/transformer_easy_mat_randomperm0_10000.pth'
load_parameter_path = None

start_epi_itr = 0

max_agent = 30
max_topic = 3


"""
for i in range(1):
    output = output_base + str(i)
    actor_weight = actor_weight_base + "_" + file_name + str(i)
    critic_weight = critic_weight_base + "_" + file_name + str(i)
    V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    train_loop_dataset(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_dir, test_data_index_dir, output)
"""


for i in range(2, 4):
    output = output_base + str(i)
    # actor_weight = actor_weight_base + "_" + file_name + str(i)
    # critic_weight = critic_weight_base + "_" + file_name + str(i)
    # V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    transformer_weight = transformer_weight_base + "_" + file_name + str(i)

    #  標準エラー出力先の変更
    sys.stderr = open(output + "_err.log", 'w')

    #happo_trainer.train_loop_single(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight,V_net_weight, learning_data_index_path, output)
    #coma_trainer.train_loop_single(max_epi_itr, buffer_size, batch_size, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_path, output)
    runner = MATRunner(max_epi_itr, batch_size, device, result_dir, backup_itr, max_agent, max_topic, learning_data_index_path=learning_data_index_path)
    runner.train_single_env_multi_process(output, transformer_weight, start_epi_itr, load_parameter_path=load_parameter_path)
    #runner.train_multi_env(output, transformer_weight, start_epi_itr, load_parameter_path=load_parameter_path)


#  標準エラー出力先を戻す
sys.stderr = sys.__stderr__
