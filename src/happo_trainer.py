from env import Env
from HAPPO import HAPPO
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import glob
import time as time_modu
import random

def read_train_curve(log_path):
    reward_history = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.split(",")

            reward_history.append(float(line[1]))
            
    return reward_history


def train_loop_single(max_epi_itr, buffer_size, batch_size, eps_clip, backup_iter, device, result_dir, actor_weight, critic_weight, learning_data_index_path, output, start_epi_itr=0, load_parameter_path=None):
    if not os.path.isdir(result_dir + "model_parameter"):
        sys.exit("結果を格納するディレクトリ" + result_dir + "model_parameter が作成されていません。")

    if start_epi_itr != 0:
        if load_parameter_path == None:
            sys.exit("読み込む重みパラメータのパスを指定してください")
        else:
            load_flag = True
    else:
        load_flag = False

    # 各種パラメーター
    N_action = 9

    #  ターゲットネットワークの同期フラグ
    target_net_flag = False

    #  何エピソードごとにターゲットネットワークを更新するか
    target_net_iter = 5

    #  標準エラー出力先の変更
    #sys.stderr = open(output + "_err.log", 'w')

    if load_flag == False:
        with open(output + ".log", 'w') as f:
            pass

    #  環境のインスタンスの生成
    env = Env(learning_data_index_path)

    episode_len = int(env.simulation_time / env.time_step)

    #  学習モデルの指定
    agent = HAPPO(N_action, env.num_client, env.num_topic, buffer_size, batch_size, episode_len, eps_clip, device)

    # 学習ループ
    for epi_iter in range(start_epi_itr, max_epi_itr):
        #  重みパラメータの読み込み
        if load_flag:
            agent.load_model(load_parameter_path, actor_weight, critic_weight, start_epi_itr)

        if epi_iter % target_net_iter == 0:
            target_net_flag = True
        else:
            target_net_flag = False
        
        #  環境のリセット
        env.reset()

        agent.old_net_update()

        #  1エピソード中の reward の保持
        reward_history = []        

        #  各エピソードにおける時間の推移
        for time in range(0, env.simulation_time, env.time_step):
            #  状態の観測
            #  obs.shape = (num_agent, num_topic, obs_channel=9, obs_size=81, obs_size=81)
            #  obs_topic.shape = (num_topic, channel=3)
            obs, obs_topic = env.get_observation()

            #  各ネットワークの入力に加工
            #  actor_obs.shape = (num_topic, num_agent, obs_channel=9, obs_size=81, obs_size=81)
            #  actor_obs_topic.shape = (num_topic, num_agent, channel=3)
            #  critic_obs.shape = (critic_obs_channel=14, obs_size=81, obs_size=81)
            #  critic_obs_topic.shape = (9)
            actor_obs, actor_obs_topic, critic_obs, critic_obs_topic = agent.process_input(obs, obs_topic)

            #  行動と確率分布の取得
            actions, pi = agent.get_acction(actor_obs, actor_obs_topic, env, train_flag=True)

            # 報酬の受け取り
            reward = env.step(actions, time)
            reward_history.append(reward)
            reward = -reward

            #  バッファへの追加
            agent.collect(actor_obs, actor_obs_topic, critic_obs, critic_obs_topic, actions, pi, reward)

        #  アドバンテージの計算
        agent.compute_advantage()

        # 学習
        agent.train(target_net_flag)

        if epi_iter % 1 == 0:
            #  ログの出力
            #print(f"total_reward = {sum(reward_history)}")
            #print(f"train is {(epi_iter/max_epi_itr)*100}% complited.")
            with open(output + ".log", 'a') as f:
                f.write(f"{(epi_iter/max_epi_itr)*100}%, {-sum(reward_history)}\n")

        #  重みパラメータのバックアップ
        if epi_iter % backup_iter == 0:
            agent.save_model(result_dir + 'model_parameter/', actor_weight, critic_weight, epi_iter)

    #  最適解の取り出し
    df_index = pd.read_csv(learning_data_index_path, index_col=0)
    opt = df_index.at['data', 'opt']

    train_curve = read_train_curve(output + ".log")

    #  学習曲線の描画
    fig = plt.figure()
    wind = fig.add_subplot(1, 1, 1)
    wind.grid()
    wind.plot(train_curve, linewidth=1, label='COMA')
    wind.axhline(y=-opt, c='r')
    fig.savefig(output + ".png")

    #  重みパラメータの保存
    agent.save_model(result_dir + 'model_parameter/', actor_weight, critic_weight, epi_iter+1)

    #  標準エラー出力先を戻す
    sys.stderr = sys.__stderr__
