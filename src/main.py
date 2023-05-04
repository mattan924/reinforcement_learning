from training_loop import *

learning_data_index_path = "../dataset/learning_data/index/index_multi.csv"
learning_data_index_dir = "../dataset/pretrain_data_set/index/"

# 各種パラメーター
max_epi_itr = 10000
buffer_size = 3000
batch_size = 500
eps_clip = 0.2
backup_iter = 500

device = "cuda:0"
result_dir = "../result/temporary/pretrain/"
output_base = result_dir + "pretrain"
actor_weight = "actor_weight"
critic_weight = "critic_weight"
V_net_weight = "V_net_weight"

for i in range(1):
    output = output_base + str(i)
    actor_weight = actor_weight + "_" + str(i)
    critic_weight = critic_weight + "_" + str(i)
    V_net_weight = V_net_weight + "_" + str(i)
    train_loop_dataset(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_dir, output, start_epi_itr=1000, load_parameter_path=result_dir+"model_parameter/")


"""
device = "cuda:0"
result_dir = "../result/COMA_fix_test/no_fix/"
output_base = result_dir + "coma_nofix"

for i in range(9):
    output = output_base + str(i)
    train_loop(max_epi_itr, device, result_dir, learning_data_index_path, output)
"""

