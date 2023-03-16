from env import Env
from COMA import COMA, ActorCritic
import matplotlib.pyplot as plt
import pandas as pd
import time as time_modu
import datetime
import sys
import os

if __name__ == '__main__':
    #  プログラムの実行時間の計測の開始
    start_time = time_modu.time()

    #  重みパラメータ、学習結果を保存するディレクトリの確認
    dt_now = datetime.datetime.now()
    date = dt_now.strftime('%m%d')
    result_dir = "../result/" + date + "/"

    if not os.path.isdir(result_dir + "model_parameter"):
        sys.exit("結果を格納するディレクトリ" + result_dir + "model_parameter が作成されていません。")

    #  標準エラー出力先の変更
    #sys.stderr = open(result_dir + "err.log", 'w')

    #  各種ログの出力先ファイルの指定
    log_file = result_dir + "out.log"
    pi_dist_file = result_dir + "pi_dist.log"

    with open(log_file, 'w') as f:
        pass

    with open(pi_dist_file, "w") as f:
        pass

    #  学習に使用するデータの指定
    learning_data_index = "../dataset/learning_data/index/index_single2.csv"

    #  環境のインスタンスの生成
    env = Env(learning_data_index)

    # 各種パラメーター
    max_epi_itr = 1000
    N_action = 9
    buffer_size = 3000
    batch_size = 500
    device = 'cuda'

    #  train_flag = True: 学習モード, False: 実行モード
    train_flag = True
    #  pretrain_flag = True: 指定の行動, False: 確率分布からサンプリング
    pretrain_flag = False
    #  load_flag = True: 重みパラメータを読み込む, False: 読み込まない
    load_flag = False
    #  学習開始時のエピソード数を指定
    start_epi_itr = 0
    #  pre_trainを行うエピソードの周期 (pre_train_iter = 10の時10回に一回 pre_train を実行)
    pre_train_iter = 10
    #  重みのバックアップを行うエピソードの周期 (backup_iter = 1000 の時1000回に一回バックアップを実行)
    backup_iter = 1000

    #  学習モデルの指定
    agent = COMA(N_action, env.num_client, buffer_size, batch_size, device)
    #agent = ActorCritic(N_action, env.num_client, env.num_topic, buffer_size, batch_size, device)

    #  学習による total_reward の推移を保存
    train_curve = []

    # 学習ループ
    for epi_iter in range(start_epi_itr, max_epi_itr):
        #  重みパラメータの読み込み
        if load_flag and (epi_iter-1) % backup_iter == 0:
            agent.load_model(result_dir + "/model_parameter/", epi_iter-1)
        
        #  環境のリセット
        env.reset()
        #  状態の観測
        obs = env.get_observation()
        next_obs = None

        #  1エピソード中の reward の保持
        reward_history = []

        #  各エピソードにおける時間の推移
        for time in range(0, env.simulation_time, env.time_step):

            # 行動の選択方式の設定
            if epi_iter % pre_train_iter == 0:
                pretrain_flag = True
            else:
                pretrain_flag = False

            #  行動と確率分布の取得
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
            #  ログの出力
            print(f"total_reward = {sum(reward_history)}")
            print(f"train is {(epi_iter/max_epi_itr)*100}% complited.")
            with open(log_file, 'a') as f:
                f.write(f"total_reward = {sum(reward_history)}\n")
                f.write(f"train is {(epi_iter/max_epi_itr)*100}% complited.\n")

            with open(pi_dist_file, "a") as f:
                for i in range(1):
                    f.write(f"agent {i} pi = {pi[i]}\n")

            if epi_iter % pre_train_iter != 0:
                train_curve.append(-sum(reward_history))

        #  重みパラメータのバックアップ
        if epi_iter % backup_iter == 0:
            agent.save_model(result_dir + 'model_parameter/', epi_iter)
            load_flag = True

    #  学習終了時の時間
    end_time = time_modu.time()

    #  計算時間の出力
    print(f"実行時間: {end_time-start_time}s")

    with open(log_file, 'a') as f:
        f.write(f"実行時間: {end_time-start_time}s\n")

    #  最適解の取り出し
    df_index = pd.read_csv(learning_data_index, index_col=0)
    opt = df_index.at['data', 'opt']

    #  学習曲線の描画
    plt.plot(train_curve, linewidth=1, label='COMA')
    plt.axhline(y=-opt, c='r')
    plt.savefig(result_dir + "reward_history.png")

    #  重みパラメータの保存
    agent.save_model(result_dir + 'model_parameter/', epi_iter+1)

    #  標準エラー出力先を戻す
    sys.stderr = sys.__stderr__
