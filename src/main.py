from training_loop import *

learning_data_index_path = "../dataset/learning_data/index/index_multi.csv"
learning_data_index_dir = "../dataset/pretrain_data_set/index/"

# 各種パラメーター
max_epi_itr = 3000
buffer_size = 3000
batch_size = 500
eps_clip = 0.2
backup_iter = 500

device = "cuda:1"
result_dir = "../result/temporary/multi_topic/ppo_check/"
output_base = result_dir + "ppo"
actor_weight = "actor_weight"
critic_weight = "critic_weight"
V_net_weight = "V_net_weight"

"""
for i in range(1,2):
    output = output_base + str(i)
    actor_weight = actor_weight + "_" + str(i)
    critic_weight = critic_weight + "_" + str(i)
    V_net_weight = V_net_weight + "_" + str(i)
    train_loop_dataset(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_dir, output)
"""


for i in range(1,5):
    output = output_base + str(i)
    actor_weight = actor_weight + "_" + str(i)
    critic_weight = critic_weight + "_" + str(i)
    V_net_weight = V_net_weight + "_" + str(i)
    train_loop_single(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_path, output)

