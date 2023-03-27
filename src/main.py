from training_loop import *


max_epi_itr = 1000
learning_data_index_path = "../dataset/learning_data/index/index_test.csv"


device = "cuda:1"
result_dir = "../result/COMA_pretrain_test/pretrain_5/"
output_base = result_dir + "pretrain"

for i in range(5,10):
    output = output_base + str(i)
    train_loop(max_epi_itr, device, result_dir, learning_data_index_path, output)


"""
device = "cuda:0"
result_dir = "../result/COMA_fix_test/no_fix/"
output_base = result_dir + "coma_nofix"

for i in range(9):
    output = output_base + str(i)
    train_loop(max_epi_itr, device, result_dir, learning_data_index_path, output)
"""

