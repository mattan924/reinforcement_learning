import happo_trainer
import coma_trainer


learning_data_index_path = "../dataset/learning_data/index/index_test.csv"
#learning_data_index_dir = "../dataset/data_set_hard/index/"

#test_data_index_dir = "../dataset/test_data_set_hard/index/"

# 各種パラメーター
#  HAPPO
max_epi_itr = 2500
buffer_size = 3000
batch_size = 300
eps_clip = 0.2
backup_iter = 100

#  COMA
#max_epi_itr = 5000
#buffer_size = 3000
#batch_size = 300
#backup_iter = 100

device = "cuda:0"
result_dir = "../result/temporary/happo_test/"
file_name = "happo_coma_learning_data_test_"
output_base = result_dir + file_name
actor_weight_base = "actor_weight"
critic_weight_base = "critic_weight"
V_net_weight_base = "V_net_weight"


"""
for i in range(1):
    output = output_base + str(i)
    actor_weight = actor_weight_base + "_" + file_name + str(i)
    critic_weight = critic_weight_base + "_" + file_name + str(i)
    V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    train_loop_dataset(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_dir, test_data_index_dir, output)
"""


for i in range(1):
    output = output_base + str(i)
    actor_weight = actor_weight_base + "_" + file_name + str(i)
    critic_weight = critic_weight_base + "_" + file_name + str(i)
    V_net_weight = V_net_weight_base + "_" + file_name + str(i)
    happo_trainer.train_loop_single(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight,V_net_weight, learning_data_index_path, output)
    #coma_trainer.train_loop_single(max_epi_itr, buffer_size, batch_size, backup_iter, device, result_dir, actor_weight, critic_weight, V_net_weight, learning_data_index_path, output)
