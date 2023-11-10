from env import Env
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os


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

"""
data_index_path = "../dataset/debug/debug/index/index_easy.csv"

log_path = "../result/temporary/debug/easy/easy_mat0.log"
log_path_tuning = "../result/temporary/parameter_tuning/easy_tuning_importance_trail89_learning_log_process4.log"
result_fig = "../result/temporary/parameter_tuning/easy_tuning_importance.png"

train_curve = read_train_curve(log_path)
train_curve_tuning = read_train_curve(log_path_tuning)*-1

df_index = pd.read_csv(data_index_path, index_col=0)
opt = df_index.at['data', 'opt']

nearest_reward = cal_nearest_server_reward(data_index_path)

fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
#wind.set_ylim(ymin=21000, ymax=34000)
wind.grid()
#wind.set_title("train iteration - total reward")
wind.set_xlabel("train iteration")
wind.set_ylabel("total reward (ms)")
wind.plot(train_curve, linewidth=1, label='mat')
wind.plot(train_curve_tuning, linewidth=1, label='tuning')
wind.axhline(y=opt, c='r', label="optimal")
wind.axhline(y=nearest_reward, c='g', label="nearest_server")
wind.legend()
fig.savefig(result_fig)
"""


flag = True

if flag:
    data_index_dir = "../dataset/similar_dataset/easy/traking_assign/test/index/"

    data_index_dir_path = os.path.join(data_index_dir, "*")
    data_index_path = natsorted(glob.glob(data_index_dir_path))

    log_path_base = "../result/temporary/similar_dataset/easy/traking_assign/hight_load_multi_noscaling0_test"
    log_scaling_path_base = "../result/temporary/similar_dataset/easy/traking_assign/hight_load_multi_scaling0_test"
    log_scaling_revision_path_base = "../result/temporary/similar_dataset/easy/traking_assign/hight_load_scaling_revision0_test"
    log_batch_path_base = "../result/temporary/similar_dataset/easy/traking_assign/hight_load_multi_scaling_batch128_0_test"

    result_fig_base = "../result/temporary/similar_dataset/easy/traking_assign/hight_load_multi"
else:
    data_index_dir = "../dataset/similar_dataset/easy/traking_assign_edge_topic/test/index/"

    data_index_dir_path = os.path.join(data_index_dir, "*")
    data_index_path = natsorted(glob.glob(data_index_dir_path))

    log_path_base = "../result/temporary/similar_dataset/easy/traking_assign_edge_topic/hight_load_multi_noscaling0_test"
    log_scaling_path_base = "../result/temporary/similar_dataset/easy/traking_assign_edge_topic/hight_load_multi_scaling0_test"
    log_scaling_revision_path_base = "../result/temporary/similar_dataset/easy/traking_assign_edge_topic/hight_load_scaling_revision0_test"
    log_batch_path_base = "../result/temporary/similar_dataset/easy/traking_assign_edge_topic/hight_load_multi_scaling_batch128_0_test"

    result_fig_base = "../result/temporary/similar_dataset/easy/traking_assign_edge_topic/hight_load_multi"


for idx in range(len(data_index_path)):
    index_path = data_index_path[idx]

    log_path = log_path_base + str(idx) + ".log"
    log_scaling_path = log_scaling_path_base + str(idx) + ".log"
    log_scaling_revision_path = log_scaling_revision_path_base + str(idx) + ".log"
    log_batch_path = log_batch_path_base + str(idx) + ".log"

    train_curve = read_train_curve(log_path)
    train_curve_scaling = read_train_curve(log_scaling_path)
    train_curve_scaling_revision = read_train_curve(log_scaling_revision_path)
    train_curve_batch = read_train_curve(log_batch_path)

    df_index = pd.read_csv(index_path, index_col=0)
    opt = df_index.at['data', 'opt']

    nearest_reward = cal_nearest_server_reward(index_path)

    fig = plt.figure()
    wind = fig.add_subplot(1, 1, 1)
    #wind.set_ylim(ymin=21000, ymax=34000)
    #wind.set_xlim(xmin=0, xmax=200)
    wind.grid()
    #wind.set_title("test " + str(i))
    wind.set_xlabel("train iteration")
    wind.set_ylabel("total reward (ms)")
    #wind.plot(train_curve, linewidth=1, label='mat')
    wind.plot(train_curve_scaling, linewidth=1, label='mat_scaling')
    #wind.plot(train_curve_scaling_revision, linewidth=1, label='mat_scaling_revision')
    #wind.plot(train_curve_batch, linewidth=1, label='mat_batch')
    wind.axhline(y=opt, c='r', label="optimal")
    wind.axhline(y=nearest_reward, c='g', label="nearest_server")
    wind.legend()
    fig.savefig(result_fig_base + str(idx) + ".png")
