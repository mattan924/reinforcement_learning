import sys
sys.path.append("../../dataset_visualization/src")
import util
import animation
from env import Env
from COMA import COMA, ActorCritic
import matplotlib.pyplot as plt
import pandas as pd
import time as time_modu
import datetime
import os

if __name__ == '__main__':
    #  学習に使用するデータの指定
    data_index = "../dataset/learning_data/index/index_test.csv"

    #  読み込む重みパラメータ
    load_parameter = "../result/COMA_test/model_parameter/"

    #  結果出力先ファイル
    output_file = "../dataset/execution_data/solution/COMA_test1000.csv"

    #  結果確認用アニメーション
    output_animation = "../dataset/execution_data/animation/COMA_test1000.gif"

    df_index = pd.read_csv(data_index, index_col=0, dtype=str)
    df_index.at['data', 'solve_file'] = output_file
    df_index.to_csv(data_index)

    with open(output_file, "w") as f:
        pass

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
    #agent = ActorCritic(N_action, env.num_client, buffer_size, batch_size, device)

    #  重みパラメータの読み込み
    agent.load_model(load_parameter, 1000)
        
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

        for i in range(env.num_client):
            client = env.pre_time_clients[i]
            util.writeSolutionCSV(output_file, client.id, time, client.x, client.y, client.pub_edge, client.sub_edge, 1)

    animation.create_single_assign_animation(data_index, output_animation, 20)