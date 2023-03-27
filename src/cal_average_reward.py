import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_train_curve(log_path, pre_train_iter):
    reward_history = []
    tmp = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.split(",")

            if tmp % pre_train_iter != 0:
                reward_history.append(float(line[1]))

            tmp += 1

        train_curve = np.zeros(len(reward_history))
        
        for i in range(len(reward_history)):
            train_curve[i] = reward_history[i]
            
    return train_curve


learning_data_index_path = "../dataset/learning_data/index/index_test.csv"

output_fix_base = "../result/COMA_fix_test/fix10/coma_fix"
output_pretrain_base = "../result/COMA_pretrain_test/pretrain_5/pretrain"

average_train_curve_fix = read_train_curve(output_fix_base + "0.log", 10)
average_train_curve_pretrain = read_train_curve(output_pretrain_base + "0.log", 5)

for i in range(1, 10):
    output_fix = output_fix_base + str(i) + ".log"
    output_pretrain = output_pretrain_base + str(i) + ".log"

    average_train_curve_fix = average_train_curve_fix + read_train_curve(output_fix, 10)
    average_train_curve_pretrain = average_train_curve_pretrain + read_train_curve(output_pretrain, 5)

average_train_curve_fix = average_train_curve_fix / 10
average_train_curve_pretrain = average_train_curve_pretrain / 10

df_index = pd.read_csv(learning_data_index_path, index_col=0)
opt = df_index.at['data', 'opt']

fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
wind.set_ylim(ymin=-40000, ymax=-25000)
wind.grid()
wind.plot(average_train_curve_fix, linewidth=1, label='pretrain_10')
wind.plot(average_train_curve_pretrain, linewidth=1, label='pretrain_5')
wind.axhline(y=-opt, c='r', label="optimal")
wind.legend()
fig.savefig("../result/COMA_pretrain_test/average_curve.png")