import pandas as pd
import numpy as np
from env import Env
import os
import sys
sys.path.append("../../dataset_visualization/src")
import util
from generator import *


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


dataset_dir = "../dataset/dataset_hard/test/"

index_file_base = dataset_dir + "index/index_hard_"
config_file = dataset_dir + "config/config_hard.csv"
traking_file_base = dataset_dir + "traking/traking_hard_"
assign_file_base = dataset_dir + "assign/assign_hard_"
edge_file_base = dataset_dir + "edge/edge_hard_"
topic_file_base = dataset_dir + "topic/topic_hard_"

max_data_size = 10

rate = 0.5

threshold = int(max_data_size * rate)

low_reward_data = 0
hight_reward_data = 0

low_data_idx = 0
hight_data_idx = threshold

while(1):
    index_file_base = dataset_dir + "index/index_hard_9.csv"
    config_file = dataset_dir + "config/config_hard.csv"
    traking_file_base = dataset_dir + "traking/traking_hard_9.csv"
    assign_file_base = dataset_dir + "assign/assign_hard_9.csv"
    edge_file_base = dataset_dir + "edge/edge_hard_9.csv"
    topic_file_base = dataset_dir + "topic/topic_hard_9.csv"

    print(f"generate")

    generate_traking(index_file_base, config_file, traking_file_base)

    generate_edge(index_file_base, config_file, edge_file_base)

    generate_topic(index_file_base, config_file, topic_file_base)

    assign_topic(index_file_base, assign_file_base)

    nearest_reward = cal_nearest_server_reward(index_file_base)

    if nearest_reward > 300:
        break

"""
while(1):
    if low_reward_data == threshold and hight_reward_data == threshold:
        break
    else:
        
        generate_traking(index_file_base, config_file, traking_file_base)

        generate_edge(index_file_base, config_file, edge_file_base)

        generate_topic(index_file_base, config_file, topic_file_base)

        assign_topic(index_file_base, assign_file_base)

        nearest_reward = cal_nearest_server_reward(index_file_base)

        if nearest_reward > 300:
            if hight_reward_data < threshold:
                index_file = index_file_base + str(hight_data_idx) + ".csv"
                traking_file =  traking_file_base + str(hight_data_idx) + ".csv"
                assign_file = assign_file_base + str(hight_data_idx) + ".csv"
                edge_file = edge_file_base + str(hight_data_idx) + ".csv"
                topic_file = topic_file_base + str(hight_data_idx) + ".csv"

                os.rename(index_file_base, index_file)
                os.rename(traking_file_base, traking_file)
                os.rename(assign_file_base, assign_file)
                os.rename(edge_file_base, edge_file)
                os.rename(topic_file_base, topic_file)

                df_index = pd.read_csv(index_file, index_col=0)
                df_index.at['data', 'traking_file'] = traking_file
                df_index.at['data', 'assign_file'] = assign_file
                df_index.at['data', 'edge_file'] = edge_file
                df_index.at['data', 'topic_file'] = topic_file

                df_index.to_csv(index_file)

                hight_reward_data += 1
                hight_data_idx += 1
        else:
            if low_reward_data < threshold:
                index_file = index_file_base + str(low_data_idx) + ".csv"
                traking_file =  traking_file_base + str(low_data_idx) + ".csv"
                assign_file = assign_file_base + str(low_data_idx) + ".csv"
                edge_file = edge_file_base + str(low_data_idx) + ".csv"
                topic_file = topic_file_base + str(low_data_idx) + ".csv"

                os.rename(index_file_base, index_file)
                os.rename(traking_file_base, traking_file)
                os.rename(assign_file_base, assign_file)
                os.rename(edge_file_base, edge_file)
                os.rename(topic_file_base, topic_file)

                df_index = pd.read_csv(index_file, index_col=0)
                df_index.at['data', 'traking_file'] = traking_file
                df_index.at['data', 'assign_file'] = assign_file
                df_index.at['data', 'edge_file'] = edge_file
                df_index.at['data', 'topic_file'] = topic_file

                df_index.to_csv(index_file)

                low_reward_data += 1
                low_data_idx += 1

        print(f"low_data = {low_reward_data}, hight_data = {hight_reward_data}")
"""