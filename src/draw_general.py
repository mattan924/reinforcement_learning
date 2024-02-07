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



num_test = 10
max_iteration = 1502


test_log_path_base = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/client20_fix20_batch16_minbatch4_0_test"
result_fig = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/low_high_client20_general_batch16.png"


test_curve = np.zeros((num_test, max_iteration+1))

for test_idx in range(num_test):
    test_log_path = test_log_path_base + str(test_idx) + ".log"

    test_curve[test_idx] = read_train_curve(test_log_path)

ave_test_curve = np.mean(test_curve, axis=0)

x = np.array(range(0, max_iteration*10+1, 10))


fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
#wind.set_ylim(ymin=100, ymax=300)
#wind.set_xlim(xmin=-100, xmax=3000)
wind.grid()
wind.set_xlabel("training iteration")
wind.set_ylabel("average total reward (ms)")
wind.plot(x, ave_test_curve, linewidth=1, label='Proposed')
#wind.axhline(y=nearest_reward, c='g', label="NS")
#wind.axhline(y=RELOC_reward, c='r', label="RELOC")
wind.legend()
fig.savefig(result_fig, bbox_inches='tight')
