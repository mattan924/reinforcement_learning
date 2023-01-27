from env import Env
from COMA_withV import COMA_withV
import matplotlib.pyplot as plt
import time as time_modu

if __name__ == '__main__':
    start_time = time_modu.time()
    learning_data_index = "../dataset/learning_data/index/index.csv"
    log_file = "../result/out.log"
    actor_grad_file = "./model_parameter/actor_grad.log"
    critic_grad_file = "./model_parameter/critic_grad.log"
    v_net_grad_file = "./model_parameter/v_net_grad.log"
    actor_file = "./model_parameter/actor.log"
    critic_file = "./model_parameter/critic.log"
    v_net_file = "./model_parameter/v_net.log"

    with open(log_file, 'w') as f:
        pass

    with open(actor_file, 'w') as f:
        pass

    with open(critic_file, 'w') as f:
        pass

    with open(v_net_file, 'w') as f:
        pass

    with open(actor_grad_file, 'w') as f:
        pass

    with open(critic_grad_file, 'w') as f:
        pass

    with open(v_net_grad_file, 'w') as f:
        pass

    env = Env(learning_data_index)

    max_epi_itr = 10
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
        with open(actor_grad_file, 'a') as f:
            f.write(f"===========================================================\n")
            f.write(f"iter = {epi_iter}\n")
            for para in agent.actor.parameters():
                f.write(f"{para.grad}\n")

        with open(critic_grad_file, 'a') as f:
            f.write(f"===========================================================\n")
            f.write(f"iter = {epi_iter}\n")
            for para in agent.critic.parameters():
                f.write(f"{para.grad}\n")

        with open(v_net_grad_file, 'a') as f:
            f.write(f"===========================================================\n")
            f.write(f"iter = {epi_iter}\n")
            for para in agent.V_net.parameters():
                f.write(f"{para.grad}\n")

        with open(actor_file, 'a') as f:
            f.write(f"===========================================================\n")
            f.write(f"iter = {epi_iter}\n")
            for para in agent.actor.parameters():
                f.write(f"{para}\n")

        with open(critic_file, 'a') as f:
            f.write(f"===========================================================\n")
            f.write(f"iter = {epi_iter}\n")
            for para in agent.critic.parameters():
                f.write(f"{para}\n")

        with open(v_net_file, 'a') as f:
            f.write(f"===========================================================\n")
            f.write(f"iter = {epi_iter}\n")
            for para in agent.V_net.parameters():
                f.write(f"{para}\n")

        # 環境のリセット
        env.reset()
        position, obs = env.get_observation()
        next_obs = None

        reward_history = []

        for time in range(0, env.simulation_time, env.time_step):

            # 行動
            actions, pi = agent.get_acction(position, obs, env.clients, train_flag)

            for i in range(10):
                print(f"agent {i} dist = {pi[i]}")

            # 報酬の受け取り
            reward = env.step(actions, time)
            reward_history.append(reward)
            reward = -reward

            # 状態の観測
            next_position, next_obs = env.get_observation()

            # 学習
            agent.train(position, obs, actions, pi, reward, next_position, next_obs)

            obs = next_obs
            position = next_position

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
    plt.savefig("../result/result.png")

    agent.save_model()