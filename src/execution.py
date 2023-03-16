from env import Env
from COMA import COMA, ActorCritic
import matplotlib.pyplot as plt
import pandas as pd
import time as time_modu
import datetime
import sys
import os

if __name__ == '__main__':
    #  学習に使用するデータの指定
    data_index = "../dataset/learning_data/index/index_single2.csv"

    #  読み込む重みパラメータ
    load_parameter = "../result/0316/model_parameter/"

    #  環境のインスタンスの生成
    env = Env(data_index)

    # 各種パラメーター
    N_action = 9
    buffer_size = 3000
    batch_size = 500
    device = 'cuda'

    #  train_flag = True: 学習モード, False: 実行モード
    train_flag = False
    #  load_flag = True: 重みパラメータを読み込む, False: 読み込まない
    load_flag = True
    #  pretrain_flag = True: 指定の行動, False: 確率分布からサンプリング
    pretrain_flag = False

    #  学習モデルの指定
    agent = COMA(N_action, env.num_client, buffer_size, batch_size, device)
    #agent = ActorCritic(N_action, env.num_client, env.num_topic, buffer_size, batch_size, device)

    #  重みパラメータの読み込み
    agent.load_model(load_parameter, 0)
        
    #  状態の観測
    obs = env.get_observation()
    next_obs = None

    #  1エピソード中の reward の保持
    reward_history = []

    #  各エピソードにおける時間の推移
    for time in range(0, env.simulation_time, env.time_step):
        #  行動と確率分布の取得
        actions, pi = agent.get_acction(obs, env, train_flag, pretrain_flag)

        # 報酬の受け取り
        reward = env.step(actions, time)
        reward_history.append(reward)
        reward = -reward

        # 状態の観測
        next_obs = env.get_observation()

        obs = next_obs

    print(f"total_reward = {sum(reward_history)}")

