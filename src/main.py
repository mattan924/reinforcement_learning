from training_loop import *

learning_data_index_path = "../dataset/learning_data/index/index_multi.csv"
learning_data_index_dir = "../dataset/pretrain_data_set/index/"

# 各種パラメーター
max_epi_itr = 3000
buffer_size = 3000
batch_size = 500
eps_clip = 0.2
backup_iter = 500

device = "cuda:0"
result_dir = "../result/temporary/multi_topic/ppo_check/"
file_name = "no_ppo"
output_base = result_dir + file_name
actor_weight_base = "actor_weight"
critic_weight_base = "critic_weight"
V_net_weight_base = "V_net_weight"

"""
for i in range(1,2):
    output = output_base + str(i)
    actor_weight = actor_weight_base + "_" + file_name + str(i)
    critic_weight = critic_weight_base + "_" + file_name + str(i)
    V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    train_loop_dataset(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_dir, output)
"""


for i in range(5):
    output = output_base + str(i)
    actor_weight = actor_weight_base + "_" + file_name + str(i)
    critic_weight = critic_weight_base + "_" + file_name + str(i)
    V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    train_loop_single(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_path, output)

