from training_loop import *


max_epi_itr = 10000
learning_data_index_path = "../dataset/learning_data/index/index_multi.csv"


device = "cuda:0"
result_dir = "../result/temporary/test/"
output_base = result_dir + "test"

for i in range(1):
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

