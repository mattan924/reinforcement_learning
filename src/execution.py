import sys
sys.path.append("../../dataset_visualization/src")
import util
import animation
from env import Env
from COMA import COMA
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    #  学習に使用するデータの指定
    data_index = "../dataset/learning_data/index/index_multi.csv"

    #  読み込む重みパラメータ
    load_parameter = "../result/temporary/multi_topic/compare_opt/model_parameter/"
    critic_weight = "critic_weight_0"
    actor_weight = "actor_weight_0"
    v_net_weight = "V_net_weight_0"

    #  結果出力先ファイル
    output_file = "../dataset/execution_data/solution/compare1000.csv"

    #  結果確認用アニメーション
    output_animation = "../dataset/execution_data/animation/compare1000.gif"

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
    eps_clip = 0.2
    device = 'cuda'

    #  train_flag = True: 学習モード, False: 実行モード
    train_flag = False
    #  load_flag = True: 重みパラメータを読み込む, False: 読み込まない
    load_flag = True
    #  pretrain_flag = True: 指定の行動, False: 確率分布からサンプリング
    pretrain_flag = False

    #  学習モデルの指定
    agent = COMA(N_action, env.num_client, env.num_topic, buffer_size, batch_size, eps_clip, device)

    #  重みパラメータの読み込み
    agent.load_model(load_parameter, actor_weight, critic_weight, v_net_weight, 1000)
        
    #  状態の観測
    obs,obs_topic = env.get_observation()
    next_obs = None
    next_obs_topic = None

    #  1エピソード中の reward の保持
    reward_history = []

    #  各エピソードにおける時間の推移
    for time in range(0, env.simulation_time, env.time_step):
        #  行動と確率分布の取得
        actions, pi, pi_old = agent.get_acction(obs, obs_topic, env, train_flag, pretrain_flag)

        # 報酬の受け取り
        reward = env.step(actions, time)
        reward_history.append(reward)
        reward = -reward

        # 状態の観測
        next_obs, next_obs_topic = env.get_observation()

        obs = next_obs
        obs_topic = next_obs_topic

        for i in range(env.num_client):
            client = env.pre_time_clients[i]
            util.write_solution_csv(output_file, client.id, time, client.x, client.y, client.pub_edge, client.sub_edge, env.num_topic)

    animation.create_assign_animation(data_index, output_animation, 20)