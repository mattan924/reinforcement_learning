from training_loop import *


max_epi_itr = 10000
learning_data_index_path = "../dataset/learning_data/index/index_multi.csv"


device = "cuda:1"
result_dir = "../result/temporary/multi_topic/target_net/"
output_base = result_dir + "target_long"
actor_weight = "actor_weight"
critic_weight = "critic_weight"
V_net_weight = "V_net_weight"

for i in range(1,2):
    output = output_base + str(i)
    actor_weight = actor_weight + "_" + str(i)
    critic_weight = critic_weight + "_" + str(i)
    V_net_weight = V_net_weight + "_" + str(i)
    train_loop(max_epi_itr, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_path, output)


"""
device = "cuda:0"
result_dir = "../result/COMA_fix_test/no_fix/"
output_base = result_dir + "coma_nofix"

for i in range(9):
    output = output_base + str(i)
    train_loop(max_epi_itr, device, result_dir, learning_data_index_path, output)
"""

