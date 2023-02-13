from env import Env
from COMA import COMA
import matplotlib.pyplot as plt
import time as time_modu
import datetime
import sys
import os

if __name__ == '__main__':
    start_time = time_modu.time()

    dt_now = datetime.datetime.now()
    date = dt_now.strftime('%m%d')
    result_dir = "../result/" + date + "/"

    if not os.path.isdir(result_dir + "model_parameter"):
        sys.exit("結果を格納するディレクトリ" + result_dir + "model_parameter が作成されていません。")

    sys.stderr = open(result_dir + "err.log", 'w')

    log_file = result_dir + "out.log"
    learning_data_index = "../dataset/learning_data/index/index.csv"
    pi_dist_file = result_dir + "pi_dist.log"
    actor_grad_file = "./model_parameter/actor_grad.log"
    critic_grad_file = "./model_parameter/critic_grad.log"
    v_net_grad_file = "./model_parameter/v_net_grad.log"
    actor_file = "./model_parameter/actor.log"
    critic_file = "./model_parameter/critic.log"
    v_net_file = "./model_parameter/v_net.log"

    with open(log_file, 'w') as f:
        pass

    with open(pi_dist_file, "w") as f:
        pass

    """
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
    """

    env = Env(learning_data_index)

    max_epi_itr = 15000
    N_action = 9
    buffer_size = 3000
    batch_size = 500
    device = 'cuda'
    train_flag = True
    pretrain_flag = False
    load_flag = False
    start_epi_itr = 0
    pre_train_iter = 10
    backup_iter = 1000

    agent = COMA(N_action, env.num_client, buffer_size, batch_size, device)

    reward_history = []
    train_curve = []

    # 学習ループ
    for epi_iter in range(start_epi_itr, max_epi_itr):
        if load_flag and epi_iter % backup_iter == 1:
            agent.load_model(result_dir + "/model_parameter/", epi_iter-1)
        
        # 環境のリセット
        env.reset()
        obs = env.get_observation()
        next_obs = None

        reward_history = []

        for time in range(0, env.simulation_time, env.time_step):

            # 行動
            if epi_iter % pre_train_iter == 0:
                pretrain_flag = True
            else:
                pretrain_flag = False

            actions, pi = agent.get_acction(obs, env, train_flag, pretrain_flag)

            # 報酬の受け取り
            reward = env.step(actions, time)
            reward_history.append(reward)
            reward = -reward

            # 状態の観測
            next_obs = env.get_observation()

            # 学習
            agent.train(obs, actions, pi, reward, next_obs)

            obs = next_obs

        if epi_iter % 1 == 0:
            print(f"total_reward = {sum(reward_history)}")
            print(f"train is {(epi_iter/max_epi_itr)*100}% complited.")
            with open(log_file, 'a') as f:
                f.write(f"total_reward = {sum(reward_history)}\n")
                f.write(f"train is {(epi_iter/max_epi_itr)*100}% complited.\n")

            with open(pi_dist_file, "a") as f:
                for i in range(10):
                    f.write(f"agent {i} pi = {pi[i]}\n")

            """
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
            """

            if epi_iter % pre_train_iter != 0:
                train_curve.append(-sum(reward_history))

        if epi_iter % backup_iter == 0:
            agent.save_model(result_dir + 'model_parameter/', epi_iter)
            load_flag = True

    end_time = time_modu.time()

    print(f"実行時間: {end_time-start_time}s")

    with open(log_file, 'a') as f:
        f.write(f"実行時間: {end_time-start_time}s\n")

    plt.plot(train_curve, linewidth=1, label='COMA')
    plt.savefig(result_dir + "reward_history.png")

    agent.save_model(result_dir + 'model_parameter/', epi_iter+1)

    sys.stderr = sys.__stderr__
