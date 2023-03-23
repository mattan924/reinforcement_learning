from training_loop import *

"""
device = "cuda:1"
result_dir = "../result/COMA_fix_test/fix/"
output_base = result_dir + "coma_fix"

"""
device = "cuda:0"
result_dir = "../result/COMA_fix_test/no_fix/"
output_base = result_dir + "coma_nofix"


max_epi_itr = 1000
learning_data_index_path = "../dataset/learning_data/index/index_test.csv"


for i in range(9):
    output = output_base + str(i)
    train_loop_fix(max_epi_itr, device, result_dir, learning_data_index_path, output)