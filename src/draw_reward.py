import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from env import Env


def read_train_curve(log_path):
    reward_history = []
    tmp = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.split(",")

            reward_history.append(float(line[1]))

            tmp += 1

        train_curve = np.zeros(len(reward_history))
        
        for i in range(len(reward_history)):
            train_curve[i] = - reward_history[i]
            
    return train_curve


def get_perm(max_agent, max_topic):
    agent_list = range(max_agent)
    topic_list = range(max_topic)

    agent_perm = list(agent_list)
    topic_perm = list(topic_list)

    return agent_perm, topic_perm


def cal_nearest_server_reward(index_path):
    nearest_reward = 0

    env = Env(index_path)
    simulation_time = env.simulation_time
    time_step = env.time_step
    num_agent = env.num_client
    num_topic = env.num_topic

    agent_perm, topic_perm = get_perm(num_agent, num_topic)

    for time in range(0, simulation_time, time_step):
        obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, obs_size=27)
        mask = np.bool_(mask.reshape(-1))
        actions = env.get_near_action(agent_perm, topic_perm)

        nearest_reward += env.step(actions[mask], agent_perm, topic_perm, time)

    return nearest_reward


low_reward_idx = []
num_low_dataset = 0
hight_reward_idx = []
num_hight_dataset = 0

for i in range(10):
    print(f"data {i} start")
    data_index_path = "../dataset/dataset_hard/train/index/index_hard_" + str(i) + ".csv"

    nearrest_reward = cal_nearest_server_reward(data_index_path)

    if nearrest_reward > 300:
        hight_reward_idx.append(i)
        num_hight_dataset += 1
    else:
        low_reward_idx.append(i)
        num_low_dataset += 1

print(f"num hight reward dataset = {num_hight_dataset}")
print(f"num low reward dataset = {num_low_dataset}")


"""
for i in range(10):
    data_index_path = "../dataset/debug/hard/test/index/index_hard_" + str(i) + ".csv"

    log_path_epoch4 = "../result/temporary/debug/hard/hard_multi_epoch4_0_test" + str(i) + ".log"
    log_path_epoch6 = "../result/temporary/debug/hard/hard_multi_epoch6_0_test" + str(i) + ".log"
    log_path_epoch8 = "../result/temporary/debug/hard/hard_multi_epoch8_0_test" + str(i) + ".log"


    train_curve4 = read_train_curve(log_path_epoch4)
    train_curve6 = read_train_curve(log_path_epoch6)
    train_curve8 = read_train_curve(log_path_epoch8)


    df_index = pd.read_csv(data_index_path, index_col=0)
    opt = df_index.at['data', 'opt']

    nearest_reward = cal_nearest_server_reward(data_index_path)

    fig = plt.figure()
    wind = fig.add_subplot(1, 1, 1)
    #wind.set_ylim(ymin=21000, ymax=34000)
    wind.grid()
    wind.set_title("test " + str(i))
    wind.plot(train_curve4, linewidth=1, label='epoch4')
    #wind.plot(train_curve6, linewidth=1, label='epoch6')
    #wind.plot(train_curve8, linewidth=1, label='epoch8')
    wind.axhline(y=opt, c='r', label="optimal")
    wind.axhline(y=nearest_reward, c='g', label="nearest_server")
    wind.legend()
    fig.savefig("../result/temporary/debug/hard/hard_multi_test" + str(i) + ".png")
"""