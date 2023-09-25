import queue
import sys
sys.path.append("../../dataset_visualization/src")
import util
import pandas as pd
import numpy as np
import math
import copy


class Client:

    def __init__(self, id, x, y, pub_topic, sub_topic, num_topic):
        self.id = id
        self.x = x
        self.y = y
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.pub_edge = np.full(num_topic, -1)
        self.sub_edge = np.full(num_topic, -1)


class Edge:

    def __init__(self, id, x, y, volume, cpu_cycle, num_topic):
        self.id = id
        self.x = x
        self.y = y
        self.max_volume = volume
        self.used_volume = np.zeros(num_topic)
        self.total_used_volume = 0
        self.deploy_topic = np.zeros(num_topic)
        self.cpu_cycle = cpu_cycle
        self.power_allocation = cpu_cycle
        self.used_publishers = np.zeros(num_topic)

    
    def cal_used_volume(self):
        self.total_used_volume = sum(self.used_volume)


class Topic:

    def __init__(self, id, save_period, publish_rate, data_size, require_cycle):
        self.id = id
        self.save_period = save_period
        self.publish_rate = publish_rate
        self.data_size = data_size
        self.require_cycle = require_cycle
        self.num_client_queue = queue.Queue()
        self.total_num_client = 0
        self.volume = 0


    #  トピックが使用するストレージ容量の計算
    def cal_volume(self, time_step):
        self.volume = self.data_size*self.publish_rate*time_step*self.total_num_client

    
    #  トピックをpublishしているクライアント数の更新
    def update_client(self, new_num_client, time_step):
        if self.num_client_queue.qsize() < self.save_period/time_step:
            self.num_client_queue.put(new_num_client)
            self.total_num_client = self.total_num_client + new_num_client
        elif self.num_client_queue.qsize() == self.save_period/time_step:
            old_num_client = self.num_client_queue.get()
            self.num_client_queue.put(new_num_client)
            self.total_num_client = self.total_num_client + new_num_client - old_num_client
        else:
            sys.exit("save_period が time_step の整数倍になっていません")


class Env:

    def __init__(self, index_file):
        #  index_file から各種ファイルパスの取り出し
        df_index = pd.read_csv(index_file, index_col=0)
        self.config_file = df_index.at['data', 'config_file']
        self.data_file = df_index.at['data', 'assign_file']
        self.edge_file = df_index.at['data', 'edge_file']
        self.topic_file = df_index.at['data', 'topic_file']

        #  config ファイルからパラメータの取り出し
        parameter = util.read_config(self.config_file)
        self.min_x = parameter['min_x']
        self.max_x = parameter['max_x']
        self.min_y = parameter['min_y']
        self.max_y = parameter['max_y']
        self.simulation_time = parameter['simulation_time']
        self.time_step = parameter['time_step']
        self.num_client = parameter['num_client']
        self.num_edge = parameter['num_edge']
        self.num_topic = parameter['num_topic']
        self.cloud_time = parameter['cloud_time']
        self.cloud_cycle = parameter['cloud_cycle']

        #  edge情報の読み込み
        edges = util.read_edge(self.edge_file)

        self.all_edge = [Edge(e.id, e.x, e.y, e.volume, e.cpu_cycle, self.num_topic) for e in edges]

        #  topic 情報の読み込み
        topics = util.read_topic(self.topic_file)

        self.all_topic = [Topic(t.id, t.save_period, t.publish_rate, t.data_size, t.require_cycle) for t in topics]

        #  pub/sub関係を持ったトラッキングデータの読み込み
        self.learning_data = util.read_data_set_topic(self.data_file, self.num_topic)

        self.publishers = [[] for _ in range(self.num_topic)]
        self.subscribers = [[] for _ in range(self.num_topic)]

        self.clients = []
        self.pre_time_clients = []
        for _ in range(self.num_client):
            data = self.learning_data.pop(0)
            client = Client(data.id, data.x, data.y, data.pub_topic, data.sub_topic, self.num_topic)

            self.clients.append(client)

            for i in range(self.num_topic):
                if client.pub_topic[i] == 1:
                    self.publishers[i].append(client)
                
                if client.sub_topic[i] == 1:
                    self.subscribers[i].append(client)
        
        #  初期状態のストレージ使用量の更新
        for topic in self.all_topic:
            topic.update_client(len(self.publishers[topic.id]), self.time_step)
            topic.cal_volume(self.time_step)


    # 環境の初期化
    def reset(self):
        #  edge情報の読み込み
        edges = util.read_edge(self.edge_file)

        self.all_edge_opt = [Edge(e.id, e.x, e.y, e.volume, e.cpu_cycle, self.num_topic) for e in edges]

        #  topic 情報の読み込み
        topics = util.read_topic(self.topic_file)

        self.all_topic_opt = [Topic(t.id, t.save_period, t.publish_rate, t.data_size, t.require_cycle) for t in topics]

        self.learning_data = util.read_data_set_topic(self.data_file, self.num_topic)

        self.clients = []
        self.publishers = [[] for _ in range(self.num_topic)]
        self.subscribers = [[] for _ in range(self.num_topic)]
        for _ in range(self.num_client):
            data = self.learning_data.pop(0)
            client = Client(data.id, data.x, data.y, data.pub_topic, data.sub_topic, self.num_topic)

            self.clients.append(client)

            for i in range(self.num_topic):
                if client.pub_topic[i] == 1:
                    self.publishers[i].append(client)
                
                if client.sub_topic[i] == 1:
                    self.subscribers[i].append(client)
        
        for topic in self.all_topic:
            topic.update_client(len(self.publishers[topic.id]), self.time_step)
            topic.cal_volume(self.time_step)
        
        
    #  状態の観測
    def get_observation(self, obs_size=81, debug=False):
        obs_channel = 9
        obs_topic_channel = 3

        #  観測値
        obs = np.zeros((self.num_client, self.num_topic, obs_channel, obs_size, obs_size))
        #  channel=0 : 各クライアントの位置
        #  channel=1 : あるトピックの publisher の分布
        #  channel=2 : あるトピックの subscriber の分布
        #  channel=3 : クライアント全体の分布
        #  channel=4 : あるトピックのストレージ使用状況
        #  channel=5 : ストレージ状況
        #  channel=6 : cpu の最大クロック数
        #  channel=7 : あるトピックの publisher がどのエッジを何人使用しているか
        #  channel=8 : どのエッジを何人使用しているか

        obs_topic = np.zeros((self.num_topic, obs_topic_channel))
        #  channel=0 : あるトピックの処理に必要なサイクル
        #  channel=1 : あるトピックのデータサイズ
        #  channel=2 : あるトピックの使用ストレージサイズ

        block_len_x = (self.max_x-self.min_x)/obs_size
        block_len_y = (self.max_y-self.min_y)/obs_size

        #  各クライアントの位置
        position_info = np.zeros((self.num_client, obs_size, obs_size))
        #  クライアント全体の分布
        total_distribution = np.zeros((obs_size, obs_size))
        #  ある topic の publisher/subscriber の分布
        publisher_distribution = np.zeros((self.num_topic, obs_size, obs_size))
        subscriber_distribution = np.zeros((self.num_topic, obs_size, obs_size))

        #  ある topic が使用しているストレージ状況
        topic_storage_info = np.zeros((self.num_topic, obs_size, obs_size))
        #  ストレージの空き状況
        storage_info = np.zeros((obs_size, obs_size))
        #  cpu の最大クロック数
        cpu_info = np.zeros((obs_size, obs_size))
        #  使用中のクライアントの数
        topic_cpu_used_client = np.zeros((self.num_topic, obs_size, obs_size))
        cpu_used_client = np.zeros((obs_size, obs_size))

        for client in self.clients:
            block_index_x = int(client.x / block_len_x)
            block_index_y = int(client.y / block_len_y)

            if block_index_x == obs_size:
                block_index_x = obs_size-1
            if block_index_y == obs_size:
                block_index_y = obs_size-1

            position_info[client.id][block_index_y][block_index_x] = 1

            for t in range(self.num_topic):
                if client.pub_topic[t] == 1:
                    publisher_distribution[t][block_index_y][block_index_x] += 1

                if client.sub_topic[t] == 1:
                    subscriber_distribution[t][block_index_y][block_index_x] += 1

            total_distribution[block_index_y][block_index_x] += 1

        for edge in self.all_edge:
            block_index_x = int(edge.x / block_len_x)
            block_index_y = int(edge.y / block_len_y)
                
            storage_info[block_index_y][block_index_x] = (edge.max_volume - edge.total_used_volume)
            cpu_info[block_index_y][block_index_x] = edge.cpu_cycle
            cpu_used_client[block_index_y][block_index_x] = sum(edge.used_publishers)

            for t in range(self.num_topic):
                topic_storage_info[t][block_index_y][block_index_x] = edge.used_volume[t]
                topic_cpu_used_client[t][block_index_y][block_index_x] = edge.used_publishers[t]

        obs[:, :, 0] = position_info[:, np.newaxis]
        obs[:, :, 1] = publisher_distribution[np.newaxis]
        obs[:, :, 2] = subscriber_distribution[np.newaxis]
        obs[:, :, 3] = total_distribution[np.newaxis, np.newaxis]
        obs[:, :, 4] = topic_storage_info[np.newaxis]
        obs[:, :, 5] = storage_info[np.newaxis, np.newaxis]
        obs[:, :, 6] = cpu_info[np.newaxis, np.newaxis]
        obs[:, :, 7] = topic_cpu_used_client[np.newaxis]
        obs[:, :, 8] = cpu_used_client[np.newaxis, np.newaxis]

        for t, topic in enumerate(self.all_topic):
            obs_topic[t, 0] = topic.require_cycle
            obs_topic[t, 1] = topic.data_size
            obs_topic[t, 2] = topic.volume

        if debug:
            print(f"position_info = {np.amax(position_info)}")
            print(f"publisher_distribution = {np.amax(publisher_distribution)}")
            print(f"subscriber_distribution = {np.amax(subscriber_distribution)}")
            print(f"total_distribution = {np.amax(total_distribution)}")
            print(f"topic_storage_info = {np.amax(topic_storage_info)}")
            print(f"storage_info = {np.amax(storage_info)}")
            print(f"cpu_info = {np.amax(cpu_info)}")
            print(f"topic_cpu_used_client = {np.amax(topic_cpu_used_client)}")
            print(f"cpu_used_client = {np.amax(cpu_used_client)}")

            print(f"topic.require_cycle  = {np.amax(obs_topic[:, 0])}")
            print(f"topic.data_size = {np.amax(obs_topic[:, 1])}")
            print(f"topic.volume = {np.amax(obs_topic[:, 2])}")

        return obs, obs_topic
    

    #  状態の観測
    def get_observation_mat(self, agent_perm, topic_perm, obs_size=81, debug=False):
        channel_dim = obs_size*obs_size

        max_agent = len(agent_perm)
        max_topic = len(topic_perm)

        #  観測値
        obs = np.zeros((max_agent, max_topic, obs_size*obs_size*9 + 3))
        #  0~80   : クライアントの位置
        #  81~161 : あるトピックの publisher の分布
        #  161~242: あるトピックの subscriber の分布
        #  243~323: クライアントの分布
        #  324~404: あるトピックが使用しているストレージ状況
        #  405~485: ストレージの空き状況
        #  486~566: CPU の最大クロック数
        #  567~647: あるトピックの publisher がどのエッジを何人使用しているか
        #  648~728: 各エッジを使用中ののクライアントの数
        #  729    : あるトピックの処理に必要なクロック数
        #  730    : 1メッセージあたりのデータサイズ
        #  731    : ストレージサイズ

        mask = np.zeros((max_agent, max_topic))

        block_len_x = (self.max_x-self.min_x)/obs_size
        block_len_y = (self.max_y-self.min_y)/obs_size

        #  各クライアントの位置
        position_info_client = np.zeros((max_agent, channel_dim))
        #  クライアント全体の分布
        total_distribution = np.zeros((channel_dim))
        #  ある topic の publisher/subscriber の分布
        publisher_distribution = np.zeros((max_topic, channel_dim))
        subscriber_distribution = np.zeros((max_topic, channel_dim))

        #  ある topic が使用しているストレージ状況
        topic_storage_info = np.zeros((max_topic, channel_dim))
        #  ストレージの空き状況
        storage_info = np.zeros((channel_dim))
        #  cpu の最大クロック数
        cpu_info = np.zeros((channel_dim))
        #  使用中のクライアントの数
        topic_cpu_used_client = np.zeros((max_topic, channel_dim))
        cpu_used_client = np.zeros((channel_dim))

        for i in range(max_agent):
            client_id = agent_perm[i]

            if client_id < self.num_client:
                client = self.clients[client_id]
                block_index_x = int(client.x / block_len_x)
                block_index_y = int(client.y / block_len_y)

                if block_index_x == obs_size:
                    block_index_x = obs_size-1
                if block_index_y == obs_size:
                    block_index_y = obs_size-1

                position_info_client[i][block_index_y*obs_size + block_index_x] = 1000

                for t in range(max_topic):
                    topic_id = topic_perm[t]
                    
                    if topic_id < self.num_topic:
                        if client.pub_topic[topic_id] == 1:
                            publisher_distribution[t][block_index_y*obs_size + block_index_x] += 1
                            mask[i][t] = 1

                        if client.sub_topic[topic_id] == 1:
                            subscriber_distribution[t][block_index_y*obs_size + block_index_x] += 1

                total_distribution[block_index_y*obs_size + block_index_x] += 1

        for edge in self.all_edge:
            block_index_x = int(edge.x / block_len_x)
            block_index_y = int(edge.y / block_len_y)

            if block_index_x == obs_size:
                block_index_x = obs_size-1
            if block_index_y == obs_size:
                block_index_y = obs_size-1
                
            storage_info[block_index_y*obs_size + block_index_x] = (edge.max_volume - edge.total_used_volume)
            cpu_info[block_index_y*obs_size + block_index_x] = edge.cpu_cycle
            cpu_used_client[block_index_y*obs_size + block_index_x] = sum(edge.used_publishers)

            for t in range(max_topic):
                topic_id = topic_perm[t]

                if topic_id < self.num_topic:
                    topic_storage_info[t][block_index_y*obs_size + block_index_x] = edge.used_volume[topic_id]
                    topic_cpu_used_client[t][block_index_y*obs_size + block_index_x] = edge.used_publishers[topic_id]

        obs[:, :, 0:channel_dim] = position_info_client[:, np.newaxis]
        obs[:, :, channel_dim:channel_dim*2] = publisher_distribution[np.newaxis]
        obs[:, :, channel_dim*2:channel_dim*3] = subscriber_distribution[np.newaxis]
        obs[:, :, channel_dim*3:channel_dim*4] = total_distribution[np.newaxis, np.newaxis]
        obs[:, :, channel_dim*4:channel_dim*5] = topic_storage_info[np.newaxis]
        obs[:, :, channel_dim*5:channel_dim*6] = storage_info[np.newaxis, np.newaxis]
        obs[:, :, channel_dim*6:channel_dim*7] = cpu_info[np.newaxis, np.newaxis]
        obs[:, :, channel_dim*7:channel_dim*8] = topic_cpu_used_client[np.newaxis]
        obs[:, :, channel_dim*8:channel_dim*9] = cpu_used_client[np.newaxis, np.newaxis]


        for t in range(self.num_topic):
            topic_id = topic_perm[t]
            if topic_id < self.num_topic:
                topic = self.all_topic[topic_id]
                obs[:, t, -3] = topic.require_cycle * 1e4
                obs[:, t, -2] = topic.data_size * 1e4
                obs[:, t, -1] = topic.volume * 1e1
            
        if debug:
            print(f"position_info = {np.amax(position_info_client)}")
            print(f"publisher_distribution = {np.amax(publisher_distribution)}")
            print(f"subscriber_distribution = {np.amax(subscriber_distribution)}")
            print(f"total_distribution = {np.amax(total_distribution)}")
            print(f"topic_storage_info = {np.amax(topic_storage_info)}")
            print(f"storage_info = {np.amax(storage_info)}")
            print(f"cpu_info = {np.amax(cpu_info)}")
            print(f"topic_cpu_used_client = {np.amax(topic_cpu_used_client)}")
            print(f"cpu_used_client = {np.amax(cpu_used_client)}")

            print(f"topic.require_cycle  = {np.amax(obs[0, :, 0])}")
            print(f"topic.data_size = {np.amax(obs[0, :, 1])}")
            print(f"topic.volume = {np.amax(obs[0, :, 2])}")

        return obs, mask
    

    def get_near_action(self, agent_perm, topic_perm):
        max_agent = len(agent_perm)
        max_topic = len(topic_perm)

        near_actions = np.ones((max_agent*max_topic, 1), dtype=np.int64)*-1

        for i in range(max_agent):
            client_id = agent_perm[i]
            if client_id < self.num_client:
                client = self.clients[client_id]

                min_idx = -1
                min_dis = 100000
                for edge in self.all_edge:
                    dis = self.cal_distance(client.x, client.y, edge.x, edge.y)

                    if dis < min_dis:
                        min_idx = edge.id
                        min_dis = dis

                for t in range(max_topic):
                    topic_id = topic_perm[t]
                    if topic_id < self.num_topic:
                        near_actions[i*max_topic + t] = min_idx

        return near_actions

 
    #  環境を進める
    def step(self, actions, agent_perm, topic_perm, time):
        actions = actions.reshape(-1).tolist()

        max_agent = len(agent_perm)
        max_topic = len(topic_perm)

        for edge in self.all_edge:
            edge.used_publishers = np.zeros(self.num_topic)

        block_len_x = (self.max_x-self.min_x)/3
        block_len_y = (self.max_y-self.min_y)/3

        for i in range(max_agent):
            agent_idx = agent_perm[i]
            if agent_idx < self.num_client:
                client = self.clients[agent_idx]

                for t in range(max_topic):
                    topic_id = topic_perm[t]
                    if topic_id < self.num_topic:

                        if client.pub_topic[topic_id] == 1:
                            client.pub_edge[topic_id] = actions.pop(0)

                            edge = self.all_edge[int(client.pub_edge[topic_id])]
                            edge.used_publishers[topic_id] += 1


                        if  client.sub_topic[topic_id] == 1:
                            block_index_x = int(client.x / block_len_x)
                            block_index_y = int(client.y / block_len_y)

                            if block_index_x == 3:
                                block_index_x = 2
                            if block_index_y == 3:
                                block_index_y = 2

                            client.sub_edge[topic_id] = block_index_y*3+block_index_x

        for edge in self.all_edge:
            edge.used_volume = np.zeros(self.num_topic)
            edge.deploy_topic = np.zeros(self.num_topic)
            num_user = edge.used_publishers.sum()

            for t in range(max_topic):
                topic_id = topic_perm[t]
                if topic_id < self.num_topic:

                    if edge.used_publishers[topic_id] > 0:
                        edge.used_volume[topic_id] = self.all_topic[topic_id].volume
                    
                    if num_user != 0:
                        edge.power_allocation = edge.cpu_cycle / num_user
                    else:
                        edge.power_allocation = edge.cpu_cycle

            edge.cal_used_volume()

            if edge.total_used_volume <= edge.max_volume:
                edge.deploy_topic = edge.used_publishers.astype(bool)
            else:
                volume = edge.total_used_volume
                cloud_topic = []
                while(volume > edge.max_volume):
                    max_id = -1
                    max_num = 0

                    for t in range(self.num_topic):
                        if not(t in cloud_topic):
                            if max_num < edge.used_publishers[t]:
                                max_id = t
                                max_num = edge.used_publishers[t]
                    
                    if max_id != -1:
                        volume -= edge.used_volume[max_id]
                        cloud_topic.append(max_id)

                for t in range(self.num_topic):
                    if edge.used_publishers[t] > 0 and not(t in cloud_topic):
                        edge.deploy_topic[t] = True
            
        # 報酬の計算
        reward = self.cal_reward()

        self.pre_time_clients = copy.deepcopy(self.clients)
    
        if time != self.simulation_time-self.time_step:
            self.clients = []
            self.publishers = [[] for _ in range(self.num_topic)]
            self.subscribers = [[] for _ in range(self.num_topic)]
            for _ in range(self.num_client):
                data = self.learning_data.pop(0)
                client = Client(data.id, data.x, data.y, data.pub_topic, data.sub_topic, self.num_topic)

                self.clients.append(client)

                for i in range(self.num_topic):
                    if client.pub_topic[i] == 1:
                        self.publishers[i].append(client)
                    
                    if client.sub_topic[i] == 1:
                        self.subscribers[i].append(client)

            for topic in self.all_topic:
                topic.update_client(len(self.publishers[topic.id]), self.time_step)
                topic.cal_volume(self.time_step)

        return reward


    # 報酬(総遅延)の計算
    def cal_reward(self):
        reward = 0
        num_message = 0
        for t in range(self.num_topic):
            for publisher in self.publishers[t]:
                for subscriber in self.subscribers[t]:
                    delay = self.cal_delay(publisher, subscriber, t)
                    reward = reward + delay
                    num_message += 1

        reward = reward / num_message
        
        return reward


    # topic n に関してある publisher からある subscriber までの遅延
    def cal_delay(self, publisher, subscriber, n):
        pub_edge = self.all_edge[int(publisher.pub_edge[n])]
        sub_edge = self.all_edge[int(subscriber.sub_edge[n])]

        delay = 0
        gamma = 0.1

        delay += gamma*self.cal_distance(publisher.x, publisher.y, pub_edge.x, pub_edge.y)
        delay += self.cal_compute_time(pub_edge, n)
        if pub_edge.deploy_topic[n]:
            delay += gamma*self.cal_distance(pub_edge.x, pub_edge.y, sub_edge.x, sub_edge.y)
        else:
            delay += 2*self.cloud_time
        delay += gamma*self.cal_distance(sub_edge.x, sub_edge.y, subscriber.x, subscriber.y)

        return delay

    
    #  処理時間の計算
    def cal_compute_time(self, edge, n):
        topic = self.all_topic[n]

        if edge.deploy_topic[n]:
            delay = (topic.require_cycle*(topic.volume / topic.data_size)) / edge.power_allocation
        else:
            delay = (topic.require_cycle*(topic.volume / topic.data_size)) / self.cloud_cycle

        return delay
    

    #  2点間の距離の計算
    def cal_distance(self, x1, y1, x2, y2):
        return ((int)(math.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))*100) / 100)
