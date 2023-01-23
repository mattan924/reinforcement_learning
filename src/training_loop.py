from env import Env
from COMA import COMA
import matplotlib.pyplot as plt
import time as time_modu

if __name__ == '__main__':
    start_time = time_modu.time()
    learning_data_index = "../dataset/learning_data/index/index.csv"
    log_file = "out.log"
    with open(log_file, 'w') as f:
        pass

    env = Env(learning_data_index)

    max_epi_itr = 10000
    N_action = 9
    device = 'cpu'

    agent = COMA(N_action, env.num_client, device)

    train_curve = []

    # 学習ループ
    for epi_iter in range(max_epi_itr):
        # 環境のリセット
        env.reset()

        obs_history = []
        actions_history = []
        pi_history = []
        reward_history = []

        for time in range(0, env.simulation_time, env.time_step):
            # 状態の観測
            start_getobs = time_modu.time()
            obs = env.get_observation()
            obs_history.append(obs)
            end_getobs = time_modu.time()

            # 行動
            start_getaction = time_modu.time()
            actions, pi = agent.get_acction(obs, env.clients)
            actions_history.append(actions)
            pi_history.append(pi)
            end_getaction = time_modu.time()

            # 報酬の受け取り
            start_step = time_modu.time()
            reward = env.step(actions, time)
            reward = -reward
            reward_history.append(reward)
            end_step = time_modu.time()

        # 学習
        agent.train(obs_history, actions_history, pi_history, reward_history)

        if epi_iter % 1 == 0:
            print(f"total_reward = {sum(reward_history)}")
            print(f"train is {(epi_iter/max_epi_itr)*100}% complited.")
            with open(log_file, 'a') as f:
                f.write(f"total_reward = {sum(reward_history)}\n")
                f.write(f"train is {(epi_iter/max_epi_itr)*100}% complited.\n")

            train_curve.append(-sum(reward_history))

    end_time = time_modu.time()

    print(f"実行時間: {end_time-start_time}s")

    with open(log_file, 'a') as f:
        f.write(f"実行時間: {end_time-start_time}s\n")

    plt.plot(train_curve, linewidth=1, label='COMA')
    plt.savefig("result.png")
    #plt.show()
