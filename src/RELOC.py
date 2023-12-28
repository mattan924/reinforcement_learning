from env import *
from scipy.sparse.linalg import svds
import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans


def get_perm(num_agent, num_topic, random_flag=True):
        
    agent_list = range(num_agent)
    topic_list = range(num_topic)

    if random_flag:
        agent_perm = random.sample(agent_list, num_agent)
        topic_perm = random.sample(topic_list, num_topic)
    else:
        agent_perm = list(agent_list)
        topic_perm = list(topic_list)

    return agent_perm, topic_perm


def update_theta(edge, all_topic, edge_topic):
    used_storage = 0
    require_cycle = 0

    for t in all_topic:
        if t.id in edge_topic:
            used_storage += t.volume
            require_cycle += t.publish_rate*edge.used_publishers[t.id]*t.require_cycle * math.log(t.volume/t.data_size)

    theta = min(edge.max_volume/(used_storage+1e-16), (edge.cpu_cycle / (require_cycle + edge.remain_cycle + 1e-16)), 1)

    return theta


def update_edge_topic(home_server, all_client, all_topic, num_edge):
    edge_topic = [[] for _ in range(num_edge)]

    for client in all_client:
        home_server_idx = int(home_server[client.id])

        for topic in all_topic:
            if client.pub_topic[topic.id] == 1 and topic.id not in edge_topic[home_server_idx]:
                edge_topic[home_server_idx].append(topic.id)

    return edge_topic    


def update_publisher_list(home_server, all_client, all_topic, num_edge):
    publisher_list = [[[] for j in range(num_edge)] for i in range(len(all_topic))]

    for client in all_client:
        home_server_idx = int(home_server[client.id])

        for topic in all_topic:
            if client.pub_topic[topic.id] == 1:
                publisher_list[topic.id][home_server_idx].append(client.id)

    return publisher_list


def change_home_server(home_server, client_id, new_edge_id, all_client, all_topic, num_edge):
    home_server[client_id] = new_edge_id

    edge_topic = update_edge_topic(home_server, all_client, all_topic, num_edge)
    publisher_list = update_publisher_list(home_server, all_client, all_topic, num_edge)

    return home_server, edge_topic, publisher_list


# client: env.Client
# edge1: env.Edge
# edge2: env.Edge
# edge1_topic: edge1 で扱う topic のリスト (topic id を格納)
# edge2_topic: edge2 で扱う topic のリスト (topic id を格納)
# publisher_list: (topic 数, エッジの数) のサイズの二次元リスト
# all_topic: env.Topic のインスタンスを全て格納したリスト
def cooperation(client, edge1, edge2, theta, edge_topic, publisher_list, all_topic):
    theta_edge1 = 0
    theta_edge2 = 0

    for topic in all_topic:
        if client.pub_topic[topic.id] == 1:
            if topic.id not in edge_topic[edge2.id]:
                edge2_topic.append(topic.id)

            publisher_list[topic.id][edge2.id].append(client.id)

            theta_edge2 = update_theta(edge2, all_topic, edge2_topic)

            if client.id in publisher_list[topic.id][edge1.id]:
                publisher_list[topic.id][edge1.id].remove(client.id)

            theta_edge1 = update_theta(edge1, all_topic, edge1_topic)


    for topic in all_topic:
        if client.pub_topic[topic.id] == 1:
            if len(publisher_list[topic.id][edge1.id]) == 0:
                edge1_topic.remove(topic.id)

                theta_edge1 = update_theta(edge1, all_topic, edge1_topic)

                
    return theta, edge_topic, publisher_list


def search_cluster(client, centroid):
    cluster_idx = -1
    min_loss = 100000000

    for idx in range(len(centroid)):
        loss = math.sqrt((centroid[idx][0] - client.x)**2 + (centroid[idx][1] - client.y)**2)

        if loss < min_loss:
            min_loss = loss
            cluster_idx = idx

    return cluster_idx


def RELOC(edge_file, all_client, all_topic, all_edge, K, M, agent_perm, topic_perm, near_actions):
    # 初期値の設定
    num_client = len(all_client)
    num_edge = len(all_edge)
    num_topic = len(all_topic)
    theta = np.ones(num_edge)
    edge_topic = [[] for _ in range(num_edge)]
    publisher_list = [[[] for i in range(num_edge)] for j in range(num_topic)]
    home_server = np.ones(num_client, dtype=np.int64)*-1
    z = np.zeros((num_client, num_client))
    actions = np.zeros((num_client, num_topic))
    client_cluster = np.ones(num_client, dtype=np.int64)

    # 不正な値の検知
    if M > num_client:
        sys.exit("引数でしていされた M が num_client を上回っています．")

    kmeans = KMeans(n_clusters=K, random_state=0)

    df = pd.read_csv(edge_file, index_col=0)
    edge_cluster = kmeans.fit_predict(df.values[:, 0:2])
    centroid = kmeans.cluster_centers_

    for idx in range(num_client):
        client_cluster[idx] = search_cluster(all_client[idx], centroid)

    # Z 行列の作成
    for client1 in all_client:
        for client2 in all_client:
            if client1.id != client2.id:
                for topic in all_topic:
                    if client1.pub_topic[topic.id] == 1:
                        if client2.pub_topic[topic.id] == 1:
                            z[client1.id][client2.id] += 1
                        
                        if client2.sub_topic[topic.id] == 1:
                            z[client1.id][client2.id] += 1


                    if client1.sub_topic[topic.id] == 1:
                        if client2.pub_topic[topic.id] == 1:
                            z[client1.id][client2.id] += 1
                        
                        if client2.sub_topic[topic.id] == 1:
                            z[client1.id][client2.id] += 1

    # 行列分解
    NUMBER_OF_FACTORS_MF = int(num_client * 0.8)
    u, s, vh = svds(z, k=NUMBER_OF_FACTORS_MF)
    s = np.diag(s)
    predicted_rations = np.dot(np.dot(u, s), vh)

    # 初期 home_server の設定
    near_actions = near_actions.reshape(num_client, num_topic)
    for i in range(num_client):
        client_id = agent_perm[i]
        home_server[client_id] = near_actions[client_id][0]

    # 各エッジで扱うトピックのアップデート
    edge_topic = update_edge_topic(home_server, all_client, all_topic, num_edge)

    # 各エッジサーバで扱う各トピックの publisher のリストを更新
    publisher_list = update_publisher_list(home_server, all_client, all_topic, num_edge)

    # θ の更新
    for i in range(num_edge):
        theta[i] = update_theta(all_edge[i], all_topic, edge_topic[i])

    # RELOC 本体
    for t in range(num_topic):
        topic_id = topic_perm[t]

        for i in range(num_client):
            client1_id = agent_perm[i]
            relation_list = []

            tmp = z[client1_id].copy()
            sorted_relation = np.sort(tmp)[::-1]

            # 関連性の強いクライアントの抽出
            for idx in range(M):
                relation_value = sorted_relation[idx]

                relation_client_id_list = np.where(tmp==relation_value)[0]

                while(True):
                    relation_client_id = np.random.choice(relation_client_id_list, 1)[0]
                    if relation_client_id not in relation_list:
                        relation_list.append(relation_client_id)
                        break

            if theta[home_server[client1_id]] < 1:
                candidate = list(np.where(theta==1)[0])

                for idx in reversed(range(len(candidate))):
                    candidate_edge_idx = candidate[idx]
                    if edge_cluster[candidate_edge_idx] != client_cluster[client1_id]:
                        del candidate[idx]

                if len(candidate) == 0:
                    break

                home_server[client1_id] = np.random.choice(candidate, 1)[0]

            # c_m の home_server の更新
            for client2_id in relation_list:
                client2_home_server = int(home_server[client2_id])
                if theta[client2_home_server] == 1 and edge_cluster[home_server[client1_id]] == client_cluster[client2_id]:
                    home_server, edge_topic, publisher_list = change_home_server(home_server, client1_id, client2_home_server, all_client, all_topic, num_edge)
                    # θ の更新
                    for i in range(num_edge):
                        theta[i] = update_theta(all_edge[i], all_topic, edge_topic[i])

                    break

            if all_client[client1_id].pub_topic[topic_id] == 1:
                client1_home_server = int(home_server[client1_id])
                if theta[client1_home_server] < 1:
                    theta_copy = theta.copy()
                    sorted_theta = np.sort(theta_copy)[::-1]

                    for idx in range(num_edge):
                        max_value = sorted_theta[idx]
                        max_resource_idx_list = list(np.where(theta_copy==max_value)[0])

                        for i in reversed(range(len(max_resource_idx_list))):
                            edge_idx = max_resource_idx_list[i]

                            if client_cluster[client1_id] != edge_cluster[edge_idx]:
                                del max_resource_idx_list[i]

                        if len(max_resource_idx_list) > 0:
                            client1_home_server = np.random.choice(max_resource_idx_list, 1)
                            break

                    home_server, edge_topic, publisher_list = change_home_server(home_server, client1_id, client1_home_server, all_client, all_topic, num_edge)

                    for i in range(num_edge):
                        theta[i] = update_theta(all_edge[i], all_topic, edge_topic)
    
    mask = np.zeros((num_client, num_topic))
    for idx in range(num_client):
        for t in range(num_topic):
            actions[idx][t] = home_server[idx]

            if all_client[idx].pub_topic[t] == 1:
                mask[idx][t] = 1

    mask = np.bool_(mask)

    return actions[mask]
