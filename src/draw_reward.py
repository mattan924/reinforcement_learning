from env import Env
from RELOC import RELOC
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
            train_curve[i] = reward_history[i]
            
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
        #obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_topic_used_storage, obs_storage, obs_cpu_cycle, obs_topic_num_used, obs_num_used, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, obs_size=27)
        obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, obs_size=27)
        mask = np.bool_(mask.reshape(-1))
        actions = env.get_near_action(agent_perm, topic_perm)

        nearest_reward += env.step(actions[mask], agent_perm, topic_perm, time)

    return nearest_reward


def cal_RELOC_reward(index_path, edge_file, K=3, M=3):
    reloc_reward = 0

    env = Env(index_path)

    for time in range(0, env.simulation_time, env.time_step):
        agent_perm, topic_perm = get_perm(env.num_client, env.num_topic)

        near_actions = env.get_near_action(agent_perm, topic_perm)

        actions = RELOC(edge_file, env.clients, env.all_topic, env.all_edge, K, M, agent_perm, topic_perm, near_actions)

        reloc_reward += env.step(actions, agent_perm, topic_perm, time)

    return reloc_reward

"""

data_index_path = "../dataset/master_thesis/single_data/index/fix_allocation_high_capacity_low_cycle.csv"

log_path = "../result/save/master_thesis/fix_allocation/high_capacity_low_cycle0.log"
result_fig = "../result/save/master_thesis/fix_allocation/high_low_training.pdf"

train_curve = read_train_curve(log_path)

df_index = pd.read_csv(data_index_path, index_col=0)
opt = df_index.at['data', 'opt']
edge_file = df_index.at['data', 'edge_file']

nearest_reward = - cal_nearest_server_reward(data_index_path)
RELOC_reward = - cal_RELOC_reward(data_index_path, edge_file, K=4, M=3)

print(f"RELOC reward = {RELOC_reward}")
print(f"nearest reward = {nearest_reward}")

fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
#wind.set_ylim(ymin=100, ymax=300)
#wind.set_xlim(xmin=4000, xmax=10000)
wind.grid()
wind.set_xlabel("training iteration")
wind.set_ylabel("total reward (ms)")
wind.plot(train_curve, linewidth=1, label='Proposed')
#wind.axhline(y=nearest_reward, c='g', label="NS")
#wind.axhline(y=RELOC_reward, c='r', label="RELOC")
wind.legend()
fig.savefig(result_fig, bbox_inches='tight')

"""



switch = "low_high40"


if switch == "high_high20":
    data_index_dir = "../dataset/master_thesis/multi_data/general_evaluation/high_capacity_high_cycle_client20_fix20_data10000/test/index/"
    data_index_dir_path = os.path.join(data_index_dir, "*")
    data_index_path = natsorted(glob.glob(data_index_dir_path))

    log_path_base = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_high_cycle/client20_fix20_batch64_minbatch1_test"

    result_fig_base = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_high_cycle/client20_fix20_test"

    for idx in range(len(data_index_path)):
        index_path = data_index_path[idx]

        log_path = log_path_base + str(idx) + ".log"

        train_curve = read_train_curve(log_path)

        df_index = pd.read_csv(index_path, index_col=0)
        opt = df_index.at['data', 'opt']
        edge_file = df_index.at['data', 'edge_file']

        nearest_reward = cal_nearest_server_reward(index_path)
        reloc_reward = cal_RELOC_reward(index_path, edge_file, K=4, M=3)

        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        #wind.set_ylim(ymin=0, ymax=2000)
        #wind.set_xlim(xmin=0, xmax=400)
        wind.grid()
        wind.set_title("test " + str(idx))
        wind.set_xlabel("training iteration")
        wind.set_ylabel("total reward")
        wind.plot(train_curve, linewidth=1, label='mat')
        #wind.axhline(y=opt, c='r', label="optimal")
        wind.axhline(y=nearest_reward, c='g', label="nearest_server")
        wind.axhline(y=reloc_reward, c='r', label="RELOC")
        wind.legend()
        fig.savefig(result_fig_base + str(idx) + ".png")

elif switch == "high_high40":
    data_index_dir = "../dataset/master_thesis/multi_data/general_evaluation/high_capacity_high_cycle_client40_fix40_data10000/test/index/"
    data_index_dir_path = os.path.join(data_index_dir, "*")
    data_index_path = natsorted(glob.glob(data_index_dir_path))

    log_path_base = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_high_cycle/client40_fix40_batch64_0_test"
    log_path_base_pretrain = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_high_cycle/client40_fix40_pretraining 0_test"

    result_fig_base = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_high_cycle/client40_fix40_test"

    for idx in range(len(data_index_path)):
        index_path = data_index_path[idx]

        log_path = log_path_base + str(idx) + ".log"
        log_path_pretrain = log_path_base_pretrain + str(idx) + ".log"

        train_curve = read_train_curve(log_path)
        train_curve_pretrain = read_train_curve(log_path_pretrain)

        df_index = pd.read_csv(index_path, index_col=0)
        opt = df_index.at['data', 'opt']
        edge_file = df_index.at['data', 'edge_file']

        nearest_reward = cal_nearest_server_reward(index_path)
        reloc_reward = cal_RELOC_reward(index_path, edge_file, K=4, M=3)

        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        #wind.set_ylim(ymin=0, ymax=2000)
        #wind.set_xlim(xmin=0, xmax=400)
        wind.grid()
        wind.set_title("test " + str(idx))
        wind.set_xlabel("training iteration")
        wind.set_ylabel("total reward")
        wind.plot(train_curve, linewidth=1, label='mat')
        wind.plot(train_curve_pretrain, linewidth=1, label='mat_pretrain')
        #wind.axhline(y=opt, c='r', label="optimal")
        wind.axhline(y=nearest_reward, c='g', label="nearest_server")
        wind.axhline(y=reloc_reward, c='r', label="RELOC")
        wind.legend()
        fig.savefig(result_fig_base + str(idx) + ".png")

elif switch == "high_low20":
    data_index_dir = "../dataset/master_thesis/multi_data/general_evaluation/high_capacity_low_cycle_client20_fix20_data100000/test/index/"
    data_index_dir_path = os.path.join(data_index_dir, "*")
    data_index_path = natsorted(glob.glob(data_index_dir_path))

    log_path_base = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_low_cycle/client20_fix20_0_test"

    result_fig_base = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_low_cycle/client20_fix20_test"

    for idx in range(len(data_index_path)):
        index_path = data_index_path[idx]

        log_path = log_path_base + str(idx) + ".log"

        train_curve = read_train_curve(log_path)

        df_index = pd.read_csv(index_path, index_col=0)
        opt = df_index.at['data', 'opt']
        edge_file = df_index.at['data', 'edge_file']

        nearest_reward = cal_nearest_server_reward(index_path)
        reloc_reward = cal_RELOC_reward(index_path, edge_file, K=4, M=3)

        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        #wind.set_ylim(ymin=0, ymax=2000)
        #wind.set_xlim(xmin=0, xmax=400)
        wind.grid()
        wind.set_title("test " + str(idx))
        wind.set_xlabel("training iteration")
        wind.set_ylabel("total reward")
        wind.plot(train_curve, linewidth=1, label='mat')
        #wind.axhline(y=opt, c='r', label="optimal")
        wind.axhline(y=nearest_reward, c='g', label="nearest_server")
        wind.axhline(y=reloc_reward, c='r', label="RELOC")
        wind.legend()
        fig.savefig(result_fig_base + str(idx) + ".png")

elif switch == "high_low40":
    data_index_dir = "../dataset/master_thesis/multi_data/general_evaluation/high_capacity_low_cycle_client40_fix40_data100000/test/index/"
    data_index_dir_path = os.path.join(data_index_dir, "*")
    data_index_path = natsorted(glob.glob(data_index_dir_path))

    log_path_base = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_low_cycle/client40_fix40_0_test"
    log_path_base_pretrain = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_low_cycle/client40_fix40_pretraining0_test"

    result_fig_base = "../result/save/master_thesis/multi_data/general_evaluation/high_capacity_low_cycle/client40_fix40_test"

    for idx in range(len(data_index_path)):
        index_path = data_index_path[idx]

        log_path = log_path_base + str(idx) + ".log"
        log_path_pretrain = log_path_base_pretrain + str(idx) + ".log"

        train_curve = read_train_curve(log_path)
        train_curve_pretrain = read_train_curve(log_path_pretrain)

        df_index = pd.read_csv(index_path, index_col=0)
        opt = df_index.at['data', 'opt']
        edge_file = df_index.at['data', 'edge_file']

        nearest_reward = cal_nearest_server_reward(index_path)
        reloc_reward = cal_RELOC_reward(index_path, edge_file, K=4, M=3)

        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        #wind.set_ylim(ymin=0, ymax=2000)
        #wind.set_xlim(xmin=0, xmax=400)
        wind.grid()
        wind.set_title("test " + str(idx))
        wind.set_xlabel("training iteration")
        wind.set_ylabel("total reward")
        wind.plot(train_curve, linewidth=1, label='mat')
        wind.plot(train_curve_pretrain, linewidth=1, label='mat_pretrainig')
        #wind.axhline(y=opt, c='r', label="optimal")
        wind.axhline(y=nearest_reward, c='g', label="nearest_server")
        wind.axhline(y=reloc_reward, c='r', label="RELOC")
        wind.legend()
        fig.savefig(result_fig_base + str(idx) + ".png")

elif switch == "low_high20":
    data_index_dir = "../dataset/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle_client20_fix20_data10000/test/index/"
    data_index_dir_path = os.path.join(data_index_dir, "*")
    data_index_path = natsorted(glob.glob(data_index_dir_path))

    log_path_base = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/client20_fix20_0_test"
    log_path_base_batch16_4 = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/client20_fix20_batch16_minbatch4_0_test"

    result_fig_base = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/client20_fix20_test"

    for idx in range(len(data_index_path)):
        index_path = data_index_path[idx]

        log_path = log_path_base + str(idx) + ".log"
        log_path_batch16_4 = log_path_base_batch16_4 + str(idx) + ".log"

        train_curve = read_train_curve(log_path)
        train_curve_batch16_4 = read_train_curve(log_path_batch16_4)

        df_index = pd.read_csv(index_path, index_col=0)
        opt = df_index.at['data', 'opt']
        edge_file = df_index.at['data', 'edge_file']

        nearest_reward = cal_nearest_server_reward(index_path)
        reloc_reward = cal_RELOC_reward(index_path, edge_file, K=4, M=3)

        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        #wind.set_ylim(ymin=0, ymax=2000)
        #wind.set_xlim(xmin=0, xmax=400)
        wind.grid()
        wind.set_title("test " + str(idx))
        wind.set_xlabel("training iteration")
        wind.set_ylabel("total reward")
        wind.plot(train_curve, linewidth=1, label='mat')
        #wind.plot(train_curve_batch16_4, linewidth=1, label='mat_batch16_4')
        #wind.axhline(y=opt, c='r', label="optimal")
        wind.axhline(y=nearest_reward, c='g', label="nearest_server")
        wind.axhline(y=reloc_reward, c='r', label="RELOC")
        wind.legend()
        fig.savefig(result_fig_base + str(idx) + ".png")

elif switch == "low_high40":
    data_index_dir = "../dataset/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle_client20_fix20_data10000/test/index/"
    data_index_dir_path = os.path.join(data_index_dir, "*")
    data_index_path = natsorted(glob.glob(data_index_dir_path))

    log_path_base_batch16 = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/client40_fix40_batch16_minbatch4_0_test"
    log_path_base_pre_batch16 = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/client40_fix40_batch16_pretraining0_test"
    log_path_base_pre_batch64 = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/client40_fix40_batch64_pretraining0_test"

    result_fig_base = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/client40_fix40_test"

    for idx in range(len(data_index_path)):
        index_path = data_index_path[idx]

        log_path_batch16 = log_path_base_batch16 + str(idx) + ".log"
        log_path_pre_batch16 = log_path_base_pre_batch16 + str(idx) + ".log"
        log_path_pre_batch64 = log_path_base_pre_batch64 + str(idx) + ".log"

        train_curve_batch16 = read_train_curve(log_path_batch16)
        train_curve_pre_batch16 = read_train_curve(log_path_pre_batch16)
        train_curve_pre_batch64 = read_train_curve(log_path_pre_batch64)

        df_index = pd.read_csv(index_path, index_col=0)
        opt = df_index.at['data', 'opt']
        edge_file = df_index.at['data', 'edge_file']

        nearest_reward =  - cal_nearest_server_reward(index_path)
        reloc_reward = - cal_RELOC_reward(index_path, edge_file, K=4, M=3)

        fig = plt.figure()
        wind = fig.add_subplot(1, 1, 1)
        #wind.set_ylim(ymin=0, ymax=2000)
        #wind.set_xlim(xmin=0, xmax=400)
        wind.grid()
        wind.set_title("test " + str(idx))
        wind.set_xlabel("training iteration")
        wind.set_ylabel("total reward")
        wind.plot(train_curve_batch16, linewidth=1, label='mat_batch16')
        wind.plot(train_curve_pre_batch16, linewidth=1, label='mat_pretraining_batch16')
        #wind.plot(train_curve_pre_batch64, linewidth=1, label='mat_pretraining_batch64')
        #wind.axhline(y=opt, c='r', label="optimal")
        wind.axhline(y=nearest_reward, c='g', label="nearest_server")
        wind.axhline(y=reloc_reward, c='r', label="RELOC")
        wind.legend()
        fig.savefig(result_fig_base + str(idx) + ".png")

