from env import Env
from COMA_withV import COMA_withV
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
    buffer_size = 3000
    batch_size = 500
    device = 'cuda'
    train_flag = True
    load_flag = False

    agent = COMA_withV(N_action, env.num_client, buffer_size, batch_size, device)

    if load_flag:
        agent.load_model()

    reward_history = []
    train_curve = []

    # 学習ループ
    for epi_iter in range(max_epi_itr):
        # 環境のリセット
        env.reset()
        obs = env.get_observation()
        next_obs = None

        reward_history = []

        for time in range(0, env.simulation_time, env.time_step):
            # 行動
            actions, pi = agent.get_acction(obs, train_flag)

            # 報酬の受け取り
            reward = env.step(actions, time)
            reward_history.append(reward)
            reward = -reward

            # 状態の観測
            next_obs = env.get_observation()

            # 学習
            agent.train(obs, actions, pi, reward, next_obs)

            obs = next_obs

        if epi_iter % 10 == 0:
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

    agent.save_model()