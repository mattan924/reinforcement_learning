from env import Env
from COMA import COMA
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os


def read_train_curve(log_path, pre_train_iter):
    reward_history = []
    tmp = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.split(",")

            if tmp % pre_train_iter != 0:
                reward_history.append(float(line[1]))

            tmp += 1
            
    return reward_history

def train_loop(max_epi_itr, device, result_dir, learning_data_index_path, output, start_epi_itr=0, load_parameter_path=None):
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
    buffer_size = 3000
    batch_size = 500
    backup_iter = 1000

    #  pretrain_flag = True: 指定の行動, False: 確率分布からサンプリング
    pretrain_flag = False

    #  pre_trainを行うエピソードの周期 (pre_train_iter = 10の時10回に一回 pre_train を実行)
    pre_train_iter = 5

    #  価値関数とActor ネットワークを交互に固定して学習するためのフラグ
    fix_net_flag = False

    #  何エピソードごとに固定する方を変えるか
    fix_net_iter = 10

    #  標準エラー出力先の変更
    sys.stderr = open(output + "_err.log", 'w')

    if load_flag == False:
        with open(output + ".log", 'w') as f:
            pass

    #  環境のインスタンスの生成
    env = Env(learning_data_index_path)

    #  学習モデルの指定
    agent = COMA(N_action, env.num_client, buffer_size, batch_size, device)

    # 学習ループ
    for epi_iter in range(start_epi_itr, max_epi_itr):
        #  重みパラメータの読み込み
        if load_flag:
            agent.load_model(load_parameter_path, start_epi_itr)

        if epi_iter % fix_net_iter == 0:
            fix_net_flag = not fix_net_flag
        
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
            actions, pi = agent.get_acction(obs, env, True, pretrain_flag)

            # 報酬の受け取り
            reward = env.step(actions, time)
            reward_history.append(reward)
            reward = -reward

            # 状態の観測
            next_obs = env.get_observation()

            # 学習
            agent.train(obs, actions, pi, reward, next_obs, fix_net_flag)

            obs = next_obs

        if epi_iter % 1 == 0:
            #  ログの出力
            #print(f"total_reward = {sum(reward_history)}")
            #print(f"train is {(epi_iter/max_epi_itr)*100}% complited.")
            with open(output + ".log", 'a') as f:
                f.write(f"{(epi_iter/max_epi_itr)*100}%, {-sum(reward_history)}\n")

        #  重みパラメータのバックアップ
        if epi_iter % backup_iter == 0:
            agent.save_model(result_dir + 'model_parameter/', epi_iter)

    #  最適解の取り出し
    df_index = pd.read_csv(learning_data_index_path, index_col=0)
    opt = df_index.at['data', 'opt']

    train_curve = read_train_curve(output + ".log", pre_train_iter)

    #  学習曲線の描画
    fig = plt.figure()
    wind = fig.add_subplot(1, 1, 1)
    wind.grid()
    wind.plot(train_curve, linewidth=1, label='COMA')
    wind.axhline(y=-opt, c='r')
    fig.savefig(output + ".png")

    #  重みパラメータの保存
    agent.save_model(result_dir + 'model_parameter/', epi_iter+1)

    #  標準エラー出力先を戻す
    sys.stderr = sys.__stderr__
