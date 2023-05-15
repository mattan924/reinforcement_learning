from training_loop import *

learning_data_index_path = "../dataset/data_set_hard/index/index_0.csv"
learning_data_index_dir = "../dataset/data_set_hard/index/"

test_data_index_dir = "../dataset/test_data_set_hard/index/"

# 各種パラメーター
max_epi_itr = 10000
buffer_size = 3000
batch_size = 500
eps_clip = 0.2
backup_iter = 100

device = "cuda:0"
result_dir = "../result/temporary/pretrain_hard/"
file_name = "ppo"
output_base = result_dir + file_name
actor_weight_base = "actor_weight"
critic_weight_base = "critic_weight"
V_net_weight_base = "V_net_weight"


for i in range(1, 2):
    output = output_base + str(i)
    actor_weight = actor_weight_base + "_" + file_name + str(i)
    critic_weight = critic_weight_base + "_" + file_name + str(i)
    V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    train_loop_dataset(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_dir, test_data_index_dir, output)


"""
for i in range(5):
    output = output_base + str(i)
    actor_weight = actor_weight_base + "_" + file_name + str(i)
    critic_weight = critic_weight_base + "_" + file_name + str(i)
    V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    train_loop_single(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_path, output)
"""
