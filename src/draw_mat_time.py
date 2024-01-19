from env import Env
from MAT.mat_runner import MATRunner
from MAT.transformer_policy import TransformerPolicy
from MAT.mat_trainer import MATTrainer
from MAT.shared_buffer import SharedReplayBuffer
from RELOC import RELOC
import numpy as np
import pandas as pd
import time as time_module
import matplotlib.pyplot as plt


def get_perm(max_agent, max_topic):
    agent_list = range(max_agent)
    topic_list = range(max_topic)

    agent_perm = list(agent_list)
    topic_perm = list(topic_list)

    return agent_perm, topic_perm


def mat_time_cal(data_index_path_base, start_num, end_num, step_num):
    print("mat_time_cal start")

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

    device = "cuda:0"
    max_topic = 3

    mat_time_history = []

    for num_publisher in range(start_num, end_num+1, step_num):
        print(f"num_publisher {num_publisher} is start")
        data_index_path = data_index_path_base + str(num_publisher) + "_fix" + str(num_publisher) + ".csv"

        max_agent = num_publisher

        runner = MATRunner(batch_size, ppo_epoch, lr, eps, weight_decay, obs_size, n_block, n_embd1, n_embd2, reward_scaling, device, max_agent, max_topic)

        env = Env(data_index_path)
        simulation_time = env.simulation_time
        time_step = env.time_step

        episode_length = int(simulation_time / time_step)

        policy = TransformerPolicy(runner.obs_dim, runner.obs_distri_dim, runner.edge_obs_size, runner.obs_info_dim, runner.N_action, runner.batch_size, runner.max_agent, runner.max_topic, runner.lr, runner.eps, runner.weight_decay, runner.n_block, runner.n_embd1, runner.n_embd2, device=runner.device, multi=False)

        trainer = MATTrainer(policy, runner.ppo_epoch, runner.device)

        buffer = SharedReplayBuffer(episode_length, runner.batch_size, runner.max_agent, runner.max_topic, runner.obs_dim, runner.N_action)

        runner.warmup(buffer, env, 0)

        reward_history = [[] for _ in range(batch_size)]

        mat_time = 0

        for time in range(0, simulation_time, time_step):
            step = int(time/time_step)

            mat_start = time_module.perf_counter()
            values_batch, actions_batch, action_log_probs_batch = runner.collect(trainer, buffer, step, deterministic=False)
            mat_end = time_module.perf_counter()

            mat_time += (mat_end - mat_start) / (simulation_time / time_step)

            reward_batch = np.zeros((batch_size), dtype=np.float32)

            agent_perm_batch = np.zeros((batch_size, max_agent), dtype=np.int64)
            topic_perm_batch = np.zeros((batch_size, max_topic), dtype=np.int64)

            obs_posi_batch = np.zeros((batch_size, max_agent, runner.obs_distri_dim), dtype=np.float32)
            obs_publisher_batch = np.zeros((batch_size, runner.max_topic, runner.obs_distri_dim), dtype=np.float32)
            obs_subscriber_batch = np.zeros((batch_size, max_topic, runner.obs_distri_dim), dtype=np.float32)
            obs_distribution_batch = np.zeros((batch_size, runner.obs_distri_dim), dtype=np.float32)
            obs_storage_batch = np.zeros((batch_size, runner.edge_obs_size), dtype=np.float32)
            obs_cpu_cycle_batch = np.zeros((batch_size, runner.edge_obs_size), dtype=np.float32)
            obs_remain_cycle_batch = np.zeros((batch_size, runner.edge_obs_size), dtype=np.float32)
            obs_topic_info_batch = np.zeros((batch_size, max_topic, runner.topic_obs_size), dtype=np.float32)
            mask_batch = np.zeros((batch_size, max_agent, max_topic), dtype=np.bool_)

            for idx in range(batch_size):
                reward = env.step(actions_batch[idx][buffer.mask[step][idx]], buffer.agent_perm[step][idx], buffer.topic_perm[step][idx], time)

                reward_history[idx].append(reward)
                if reward_scaling == True:
                    reward_batch[idx] = (-reward / 200) + 1
                else:
                    reward_batch[idx] = (-reward)

                agent_perm, topic_perm = runner.get_perm(random_flag=runner.random_flag)
                agent_perm_batch[idx] = agent_perm
                topic_perm_batch[idx] = topic_perm

                obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask = env.get_observation_mat(agent_perm, topic_perm, runner.obs_size)
                obs_posi_batch[idx] = obs_posi
                obs_publisher_batch[idx] = obs_publisher
                obs_subscriber_batch[idx] = obs_subscriber
                obs_distribution_batch[idx] = obs_distribution
                obs_storage_batch[idx] = obs_storage
                obs_cpu_cycle_batch[idx] = obs_cpu_cycle
                obs_remain_cycle_batch[idx] = obs_remain_cycle
                obs_topic_info_batch[idx] = obs_topic_info
                mask_batch[idx] = mask

            runner.insert_batch(buffer, obs_posi_batch, obs_publisher_batch, obs_subscriber_batch, obs_distribution_batch, obs_storage_batch, obs_cpu_cycle_batch, obs_remain_cycle_batch, obs_topic_info_batch, mask_batch, reward_batch, values_batch, actions_batch, action_log_probs_batch, agent_perm_batch, topic_perm_batch)

        mat_time_history.append(mat_time)
        print(f"mat_time = {mat_time}")

    return mat_time_history


def RELOC_time_cal(data_index_path_base, start_num, end_num, step_num, K=3):
    print(f"RELOC_time_cal start")
    reloc_time_history = []

    for num_publisher in range(start_num, end_num+1, step_num):
        print(f"num_publisher {num_publisher} start")
        data_index_path = data_index_path_base + str(num_publisher) + "_fix" + str(num_publisher) + ".csv"

        df_index = pd.read_csv(data_index_path, index_col=0)
        edge_file = df_index.at['data', 'edge_file']

        reloc_time = 0

        env = Env(data_index_path)

        M = int(num_publisher / 3)

        for time in range(0, env.simulation_time, env.time_step):
            agent_perm, topic_perm = get_perm(env.num_client, env.num_topic)

            near_actions = env.get_near_action(agent_perm, topic_perm)

            reloc_start = time_module.perf_counter()
            actions = RELOC(edge_file, env.clients, env.all_topic, env.all_edge, K, M, agent_perm, topic_perm, near_actions)
            reloc_end = time_module.perf_counter()

            reloc_time += (reloc_end - reloc_start) / (env.simulation_time / env.time_step)

            reloc_reward = env.step(actions, agent_perm, topic_perm, time)

        reloc_time_history.append(reloc_time)
        print(f"reloc time = {reloc_time}")

    return reloc_time_history


data_index_path_base = "../dataset/master_thesis/single_data/index/mat_time_client"
result_fig_path = "../result/save/master_thesis/mat_time/mat_time.png"

start_num = 10
end_num = 100
step_num = 10

mat_time_history = mat_time_cal(data_index_path_base, start_num, end_num, step_num)
reloc_time_history = RELOC_time_cal(data_index_path_base, start_num, end_num, step_num, K=3)

width = 3.5

mat_x = np.arange(start_num, end_num+1, step_num) - width/2
mat_y = np.array(mat_time_history)

reloc_x = np.arange(start_num, end_num+1, step_num) + width/2
reloc_y = np.array(reloc_time_history)

labels = list(range(start_num, end_num+1, step_num))

print(f"mat_x = {mat_x}")
print(f"reloc_x = {reloc_x}")

fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
# wind.grid(which="major", lw=0.7)
# wind.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
# wind.grid(which="minor", lw=0.7)
wind.grid(axis="y")
wind.set_title("Impact of the number of publishers on the solving time")
wind.set_xlabel("number of publishers")
wind.set_ylabel("solving time (s)")
wind.set_xticks(labels)
wind.bar(mat_x, mat_y, width=width, label="MAT")
wind.bar(reloc_x, reloc_y, width=width, label="RELOC")
wind.legend()
fig.savefig(result_fig_path)