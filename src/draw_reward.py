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
        obs, mask = env.get_observation_mat(agent_perm, topic_perm)
        mask = np.bool_(mask.reshape(-1))
        actions = env.get_near_action(agent_perm, topic_perm)

        nearest_reward += env.step(actions[mask], agent_perm, topic_perm, time)

    return nearest_reward



data_index_path = "../dataset/debug/onetopic/test/index/index_onetopic_8.csv"

log_path = "../result/temporary/debug/onetopic/onetopic_multi0_test1.log"

train_curve = read_train_curve(log_path)

df_index = pd.read_csv(data_index_path, index_col=0)
opt = df_index.at['data', 'opt']

nearest_reward = cal_nearest_server_reward(data_index_path)

fig = plt.figure()
wind = fig.add_subplot(1, 1, 1)
#wind.set_ylim(ymin=21000, ymax=34000)
wind.grid()
wind.set_title("test 8")
wind.plot(train_curve, linewidth=1, label='mat')
wind.axhline(y=opt, c='r', label="optimal")
wind.axhline(y=nearest_reward, c='g', label="nearest_server")
wind.legend()
fig.savefig("../result/temporary/debug/onetopic/onetopic_multi0_test8.png")