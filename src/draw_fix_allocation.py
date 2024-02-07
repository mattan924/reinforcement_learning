from env import Env
from RELOC import RELOC
from MAT.mat_runner import MATRunner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import sys
sys.path.append("../../dataset_visualization/src/")
import util


def get_perm(max_agent, max_topic):
    agent_list = range(max_agent)
    topic_list = range(max_topic)

    agent_perm = list(agent_list)
    topic_perm = list(topic_list)

    return agent_perm, topic_perm


def cal_nearest_server_reward(index_path, fixed=False):
    nearest_reward_history = []

    env = Env(index_path)
    simulation_time = env.simulation_time
    time_step = env.time_step
    num_agent = env.num_client
    num_topic = env.num_topic

    agent_perm, topic_perm = get_perm(num_agent, num_topic)

    for time in range(0, simulation_time, time_step):
        if fixed == True:
            if time == 0:
                obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, obs_size=27)
                mask = np.bool_(mask.reshape(-1))

                actions = env.get_near_action(agent_perm, topic_perm)
        else:
            obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, obs_size=27)
            mask = np.bool_(mask.reshape(-1))

            actions = env.get_near_action(agent_perm, topic_perm)

        nearest_reward = env.step(actions[mask], agent_perm, topic_perm, time)
        nearest_reward_history.append(nearest_reward)

    return nearest_reward_history


def cal_RELOC_reward(index_path, fixed=False, K=3, M=3):
    df_index = pd.read_csv(index_path, index_col=0)
    edge_file = df_index.at['data', 'edge_file']

    reloc_reward_history = []

    env = Env(index_path)

    for time in range(0, env.simulation_time, env.time_step):
        if fixed == True:
            if time == 0:
                agent_perm, topic_perm = get_perm(env.num_client, env.num_topic)

                near_actions = env.get_near_action(agent_perm, topic_perm)

                actions = RELOC(edge_file, env.clients, env.all_topic, env.all_edge, K, M, agent_perm, topic_perm, near_actions)
        else:
            agent_perm, topic_perm = get_perm(env.num_client, env.num_topic)

            near_actions = env.get_near_action(agent_perm, topic_perm)

            actions = RELOC(edge_file, env.clients, env.all_topic, env.all_edge, K, M, agent_perm, topic_perm, near_actions)

        reloc_reward = env.step(actions, agent_perm, topic_perm, time)
        reloc_reward_history.append(reloc_reward)

    return reloc_reward_history


def cal_MAT_reward(index_path, load_parameter_path):
    obs_size = 9
    batch_size = 1
    ppo_epoch = 6
    lr = 0.0005
    eps = 1e-05
    weight_decay = 0.0001
    n_block = 3
    n_embd1 = 81
    n_embd2 = 9
    reward_scaling = False

    num_mini_batch = 1

    device = "cuda:1"

    max_agent = 40
    max_topic = 3

    runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd1, n_embd2, reward_scaling, num_mini_batch, device, max_agent, max_topic)

    mat_reward_histoory = runner.execute_single_env(index_path, load_parameter_path)

    return mat_reward_histoory




data_index_path = "../dataset/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle_client40_fix40_data10000/test/index/data_fix_traking15_assign8_edge0_topic3.csv"
result_fig_path = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/low_high_client40_general.pdf"

load_parameter_path = "../result/save/master_thesis/multi_data/general_evaluation/low_capacity_high_cycle/model_parameter/transformer_client20_fix20_batch16_minbatch4_0_10000.pth"

fixed = False

nearest_server_reward_history = cal_nearest_server_reward(data_index_path, fixed=fixed)
reloc_reward_history = cal_RELOC_reward(data_index_path, fixed=fixed, K=3, M=1)
mat_reward_histoory = cal_MAT_reward(data_index_path, load_parameter_path)[0]


df = pd.read_csv(data_index_path, index_col=0)
config_file = df.at['data', 'config_file']
parameter = util.read_config(config_file)

time_step = parameter['time_step']
simulation_time = parameter['simulation_time']

time_data = np.arange(0, simulation_time, time_step)

fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
#wind.set_ylim(ymin=0, ymax=13)
wind.grid(which="major", lw=0.7)
wind.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
wind.grid(which="minor", lw=0.7)
# wind.set_title("Impact of fixed allocation on delays")
wind.set_xlabel("time (s)")
wind.set_ylabel("average delay (ms)")
wind.plot(time_data, nearest_server_reward_history, linewidth=1, label='NS')
wind.plot(time_data, reloc_reward_history, linewidth=1, label='RELOC')
wind.plot(time_data, mat_reward_histoory, linewidth=1, label='Proposed')
wind.legend()
fig.savefig(result_fig_path)



"""

data_index_path = "../dataset/master_thesis/single_data/index/fix_allocation_low_capacity_high_cycle.csv"
result_fig_path = "../result/save/master_thesis/fix_allocation/low_high_dynamic.pdf"

load_parameter_path = "../result/save/master_thesis/fix_allocation/model_parameter/transformer_low_capacity_high_cycle_batch16_0_5000.pth"

fixed = False

nearest_server_reward_history = cal_nearest_server_reward(data_index_path, fixed=fixed)
reloc_reward_history = cal_RELOC_reward(data_index_path, fixed=fixed, K=3, M=1)
mat_reward_histoory = cal_MAT_reward(data_index_path, load_parameter_path)[0]


df = pd.read_csv(data_index_path, index_col=0)
config_file = df.at['data', 'config_file']
parameter = util.read_config(config_file)

time_step = parameter['time_step']
simulation_time = parameter['simulation_time']

time_data = np.arange(0, simulation_time, time_step)

fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
wind.set_ylim(ymin=0, ymax=300)
wind.grid(which="major", lw=0.7)
wind.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
wind.grid(which="minor", lw=0.7)
# wind.set_title("Impact of fixed allocation on delays")
wind.set_xlabel("time (s)")
wind.set_ylabel("average delay (ms)")
wind.plot(time_data, nearest_server_reward_history, linewidth=1, label='NS')
wind.plot(time_data, reloc_reward_history, linewidth=1, label='RELOC')
wind.plot(time_data, mat_reward_histoory, linewidth=1, label='Proposed')
wind.legend()
fig.savefig(result_fig_path)

"""