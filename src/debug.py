import sys
sys.path.append("../../dataset_visualization/src")
import util
from env import Env
import pandas as pd
import numpy as np

if __name__ == '__main__':
    #  検証に使用するデータの index のパス
    data_index = "../dataset/learning_data/index/index_multi.csv"

    env = Env(data_index)

    #  使用する割り当てファイルのパス
    solution_file_opt = "../dataset/learning_data/solution/solution_multi.csv"
    solution_file = "../dataset/execution_data/solution/no_ppo_3_3000.csv"

    opt_assign = util.read_data_set_solution(solution_file_opt, env.num_topic)

    reward_history = []

    for time in range(0, env.simulation_time, env.time_step):
        actions = np.full((env.num_topic, env.num_client), -1)

        for i in range(env.num_client):
            data = opt_assign.pop(0)

            for t in range(env.num_topic):
                actions[t][i] = data.pub_edge[t]

        reward = env.step(actions, time)

        reward_history.append(reward)

    print(f"total_reward = {sum(reward_history)}")