import queue
import sys
sys.path.append("../../dataset_visualization/src")
import util
import pandas as pd
import numpy as np
import math


class Client:

    def __init__(self, id, x, y, pub_topic, sub_topic, num_topic):
        self.id = id
        self.x = x
        self.y = y
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.pub_edge = np.ones(num_topic)*-1
        self.sub_edge = -1


class Edge:

    def __init__(self, id, x, y, volume, cpu_power, num_topic):
        self.id = id
        self.x = x
        self.y = y
        self.max_volume = volume
        self.used_volume = np.zeros(num_topic)
        self.total_used_volume = 0
        self.deploy_topic = np.zeros(num_topic)
        self.cpu_power = cpu_power
        self.power_allocation = np.zeros(num_topic)
        self.used_publisers = np.zeros(num_topic)

    
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
        self.all_edge = []
        for e in edges:
            edge = Edge(e.id, e.x, e.y, e.volume, e.cpu_power, self.num_topic)
            self.all_edge.append(edge)

        #  topic 情報の読み込み
        topics = util.read_topic(self.topic_file)
        self.all_topic = []
        for t in topics:
            topic = Topic(t.id, t.save_period, t.publish_rate, t.data_size, t.require_cycle)
            self.all_topic.append(topic)

        #  pub/sub関係を持ったトラッキングデータの読み込み
        self.learning_data = util.read_data_set_topic(self.data_file, self.num_topic)

        self.clients = []
        self.pre_time_clients = []
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
        
        #  初期状態のストレージ使用量の更新
        for topic in self.all_topic:
            topic.update_client(len(self.publishers[topic.id]), self.time_step)
            topic.cal_volume(self.time_step)


    # 環境の初期化
    def reset(self):
        edges = util.read_edge(self.edge_file)
        self.all_edge = []
        for e in edges:
            edge = Edge(e.id, e.x, e.y, e.volume, e.cpu_power, self.num_topic)
            self.all_edge.append(edge)

        topics = util.read_topic(self.topic_file)
        self.all_topic = []
        for t in topics:
            topic = Topic(t.id, t.save_period, t.publish_rate, t.data_size, t.require_cycle)
            self.all_topic.append(topic)

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
    def get_observation(self):
        obs_channel = 8
        obs_size = 81

        #  観測値
        obs = np.zeros((self.num_client, self.num_topic, obs_channel, obs_size, obs_size))

        block_len_x = (self.max_x-self.min_x)/obs_size
        block_len_y = (self.max_y-self.min_y)/obs_size

        #  各クライアントの位置
        position_info = np.zeros((self.num_client, obs_size, obs_size))
        #  クライアント全体の分布
        total_distribution = np.zeros((obs_size, obs_size))
        #  ある topic の publisher/subscriber の分布
        publisher_distribution = np.zeros((self.num_topic, obs_size, obs_size))
        subscriber_distribution = np.zeros((self.num_topic, obs_size, obs_size))

        for client in self.clients:
            block_index_x = int(client.x / block_len_x)
            block_index_y = int(client.y / block_len_y)

            if block_index_x == obs_size:
                block_index_x = obs_size-1
            if block_index_y == obs_size:
                block_index_y = obs_size-1

            position_info[client.id][block_index_y][block_index_x] = 100

            for t in range(self.num_topic):
                if client.pub_topic[t] == 1:
                    publisher_distribution[t][block_index_y][block_index_x] += 1

                if client.sub_topic[t] == 1:
                    subscriber_distribution[t][block_index_y][block_index_x] += 1

            total_distribution[block_index_y][block_index_x] += 1

        #  ある topic が使用しているストレージ状況
        topic_storage_info = np.zeros((self.num_topic, obs_size, obs_size))
        #  ストレージの空き状況
        storage_info = np.zeros((obs_size, obs_size))
        #  cpu の最大クロック数
        cpu_info = np.zeros((obs_size, obs_size))
        #  使用中のクライアントの数
        cpu_used_client = np.zeros((obs_size, obs_size))

        for edge in self.all_edge:
            block_index_x = int(edge.x / block_len_x)
            block_index_y = int(edge.y / block_len_y)
                
            storage_info[block_index_y][block_index_x] = (edge.max_volume - edge.total_used_volume) / 1e5
            cpu_info[block_index_y][block_index_x] = edge.cpu_power
            cpu_used_client[block_index_y][block_index_x] = sum(edge.used_publishers)

            for t in range(self.num_topic):
                topic_storage_info[t][block_index_y][block_index_x] = edge.used_volume[t]

        for i in range(self.num_client):
            for t in range(self.num_topic):
                obs[i][t][0] = position_info[i]
                obs[i][t][1] = publisher_distribution[t]
                obs[i][t][2] = subscriber_distribution[t]
                obs[i][t][3] = total_distribution
                obs[i][t][4] = topic_storage_info[t]
                obs[i][t][5] = storage_info
                obs[i][t][6] = cpu_info
                obs[i][t][7] = cpu_used_client

        obs_topic_channel = 3
        obs_topic = np.zeros(self.num_topic, obs_topic_channel)

        for t in range(self.num_topic):
            topic = self.all_topic[t]
            obs_topic[t][0] = topic.require_cycle
            obs_topic[t][1] = topic.data_size
            obs_topic[t][2] = topic.volume

        return obs, obs_topic
        
 
    #  環境を進める
    def step(self, actions, time):
        for edge in self.all_edge:
            edge.used_publishers = np.zeros(self.num_topic)

        block_len_x = (self.max_x-self.min_x)/3
        block_len_y = (self.max_y-self.min_y)/3

        for t in range(self.num_topic):
            for publisher in self.publishers[t]:
                publisher.pub_edge[t] = actions[t][publisher.id]

                edge = self.all_edge[int(publisher.pub_edge[0])]
                edge.used_publishers[t] += 1

            for subscriber in self.subscribers[t]:
                block_index_x = int(subscriber.x / block_len_x)
                block_index_y = int(subscriber.y / block_len_y)

                if block_index_x == 3:
                    block_index_x = 2
                if block_index_y == 3:
                    block_index_y = 2

                subscriber.sub_edge = block_index_y*3+block_index_x

        for edge in self.all_edge:
            edge.used_volume = np.zeros(self.num_topic)
            edge.deploy_topic = np.zeros(self.num_topic)
            tmp = 0

            for t in range(self.num_topic):
                if len(edge.used_publishers[t]) > 0:
                    edge.used_volume[t] = self.all_topic[t].volume
                    tmp += 1
                
                if tmp != 0:
                    if len(edge.used_publishers[t]) != 0:
                        edge.power_allocation[t] = (edge.cpu_power / tmp) / len(edge.used_publishers[t])
                    else:
                        edge.power_allocation[t] = edge.cpu_power / tmp
                else:
                    edge.power_allocation[t] = edge.cpu_power

            edge.cal_used_volume()

            if edge.total_used_volume <= edge.max_volume:
                for t in range(self.num_topic):
                    if len(edge.used_publishers[t]) > 0:
                        edge.deploy_topic[t] = True
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
                        cloud_topic.append[max_id]

                for t in range(self.num_topic):
                    if len(edge.used_publishers[t]) > 0 and not(t in cloud_topic):
                        edge.deploy_topic[t] = True
            
        # 報酬の計算
        reward = self.cal_reward()

        self.pre_time_clients = self.clients
    
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
        for t in range(self.num_topic):
            for publisher in self.publishers[t]:
                for subscriber in self.subscribers[t]:
                    delay = self.cal_delay(publisher, subscriber, t)
                    reward = reward + delay
                
        return reward


    # topic n に関してある publisher からある subscriber までの遅延
    def cal_delay(self, publisher, subscriber, n):
        pub_edge = self.all_edge[int(publisher.pub_edge[n])]
        sub_edge = self.all_edge[int(subscriber.sub_edge)]

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
            delay = (topic.require_cycle*(topic.volume / topic.data_size)) / edge.power_allocation[n]
        else:
            delay = (topic.require_cycle*(topic.volume / topic.data_size)) / self.cloud_cycle

        return delay
    

    #  2点間の距離の計算
    def cal_distance(self, x1, y1, x2, y2):
        return ((int)(math.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))*100) / 100)
