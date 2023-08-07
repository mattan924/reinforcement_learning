import happo_trainer
import coma_trainer
from MAT.mat_runner import MATRunner
import sys


learning_data_index_path = "../dataset/debug/index/index_easy.csv"
#learning_data_index_dir = "../dataset/data_set_hard/index/"

#test_data_index_dir = "../dataset/test_data_set_hard/index/"

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
max_epi_itr = 10000
batch_size = 16
backup_itr = 100


device = "cuda:1"
result_dir = "../result/temporary/debug/"
file_name = "easy_mat_fixperm"
output_base = result_dir + file_name
# actor_weight_base = "actor_weight"
# critic_weight_base = "critic_weight"
# V_net_weight_base = "V_net_weight"
transformer_weight_base = "transformer"

start_epi_itr = 0


"""
for i in range(1):
    output = output_base + str(i)
    actor_weight = actor_weight_base + "_" + file_name + str(i)
    critic_weight = critic_weight_base + "_" + file_name + str(i)
    V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    train_loop_dataset(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_dir, test_data_index_dir, output)
"""


for i in range(2):
    output = output_base + str(i)
    # actor_weight = actor_weight_base + "_" + file_name + str(i)
    # critic_weight = critic_weight_base + "_" + file_name + str(i)
    # V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    transformer_weight = transformer_weight_base + "_" + file_name + str(i)

    #  標準エラー出力先の変更
    sys.stderr = open(output + "_err.log", 'w')

    #happo_trainer.train_loop_single(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight,V_net_weight, learning_data_index_path, output)
    #coma_trainer.train_loop_single(max_epi_itr, buffer_size, batch_size, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_path, output)
    runner = MATRunner(max_epi_itr, batch_size, device, result_dir, backup_itr, learning_data_index_path)
    runner.train_loop_single(output, transformer_weight, start_epi_itr)


#  標準エラー出力先を戻す
sys.stderr = sys.__stderr__
