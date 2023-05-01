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


def read_train_curve_design(log_path):
    reward_history = []
    tmp = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.split(",")

            reward = float(line[1])

            if reward != -2939134.091680464:
                reward_history.append(reward)

        train_curve = np.zeros(len(reward_history))
        
        for i in range(len(reward_history)):
            train_curve[i] = reward_history[i]
            
    return train_curve



learning_data_index_path = "../dataset/learning_data/index/index_multi.csv"

output_fix_base = "../result/temporary/multi_topic/target_net/target_pretrain"

output_fix = output_fix_base + "0.log"

train_curve = read_train_curve_design(output_fix)

#average_train_curve_fix = average_train_curve_fix / 10

df_index = pd.read_csv(learning_data_index_path, index_col=0)
opt = df_index.at['data', 'opt']

fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
#wind.set_ylim(ymin=-40000, ymax=-25000)
wind.grid()
wind.plot(train_curve, linewidth=1, label='multi_topic')
wind.axhline(y=-opt, c='r', label="optimal")
wind.legend()
fig.savefig("../result/temporary/multi_topic/target_net/target_pretrain_0.png")