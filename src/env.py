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
        self.used_volume = 0
        self.cpu_power = cpu_power
        self.power_allocation = np.zeros(num_topic)
        self.used_publisers = [[] for _ in range(num_topic)]


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
    #  マルチトピックへ改修が必要
    def get_observation(self):
        obs_channel = 4
        obs_size = 81
        obs = np.zeros((self.num_client, obs_channel, obs_size, obs_size))

        block_len_x = (self.max_x-self.min_x)/obs_size
        block_len_y = (self.max_y-self.min_y)/obs_size

        distribution = np.zeros((obs_size, obs_size))

        for client in self.clients:
            block_index_x = int(client.x / block_len_x)
            block_index_y = int(client.y / block_len_y)

            if block_index_x == obs_size:
                block_index_x = obs_size-1
            if block_index_y == obs_size:
                block_index_y = obs_size-1

            distribution[block_index_y][block_index_x] += 1
        
        storage_info = np.zeros((obs_size, obs_size))
        cpu_info = np.zeros((obs_size, obs_size))

        for edge in self.all_edge:
            block_index_x = int(edge.x / block_len_x)
            block_index_y = int(edge.y / block_len_y)

            storage_info[block_index_y][block_index_x] = (edge.max_volume - edge.used_volume) / 1e5
            cpu_info[block_index_y][block_index_x] = edge.power_allocation[0] / 1e8


        for i in range(self.num_client):
            position_info = np.zeros((obs_size, obs_size))
            client = self.clients[i]

            block_index_x = int(client.x / block_len_x)
            block_index_y = int(client.y / block_len_y)

            if block_index_x == obs_size:
                block_index_x = obs_size-1
            if block_index_y == obs_size:
                block_index_y = obs_size-1
                
            position_info[block_index_y][block_index_x] = 100
            obs[i][0] = position_info
            obs[i][1] = distribution
            obs[i][2] = storage_info
            obs[i][3] = cpu_info

        return obs
        
 
    #  環境を進める
    def step(self, actions, time):
        for edge in self.all_edge:
            edge.used_publishers = [[] for _ in range(self.num_topic)]

        block_len_x = (self.max_x-self.min_x)/3
        block_len_y = (self.max_y-self.min_y)/3

        for n in range(self.num_topic):
            for publisher in self.publishers[n]:
                publisher.pub_edge[n] = actions[publisher.id][n]

                edge = self.all_edge[int(publisher.pub_edge[n])]
                edge.used_publishers[n].append(publisher)

        for n in range(self.num_topic):
            for subscriber in self.subscribers[n]:
                block_index_x = int(subscriber.x / block_len_x)
                block_index_y = int(subscriber.y / block_len_y)

                if block_index_x == 3:
                    block_index_x = 2
                if block_index_y == 3:
                    block_index_y = 2

                subscriber.sub_edge = block_index_y*3+block_index_x

        for edge in self.all_edge:
            edge.used_volume = 0
            tmp = 0

            for n in range(self.num_topic):
                if len(edge.used_publishers[n]) > 0:
                    edge.used_volume += self.all_topic[n].volume
                    tmp += 1
                
            for n in range(self.num_topic):
                if tmp != 0:
                    edge.power_allocation[n] = (edge.cpu_power / tmp) / len(edge.used_publishers[n])
                else:
                    edge.power_allocation[n] = edge.cpu_power
            
        # 報酬の計算
        reward = self.cal_reward()
    
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
        for n in range(self.num_topic):
            for publisher in self.publishers[n]:
                for subscriber in self.subscribers[n]:
                    delay = self.cal_delay(publisher, subscriber, n)
                    reward = reward + delay
                
        return reward


    # topic n に関してある publisher からある subscriber までの遅延
    def cal_delay(self, publisher, subscriber, n):
        pub_edge = self.all_edge[int(publisher.pub_edge[n])]
        sub_edge = self.all_edge[int(subscriber.sub_edge[n])]

        delay = 0
        gamma = 0.1

        delay += gamma*self.cal_distance(publisher.x, publisher.y, pub_edge.x, pub_edge.y)
        delay += self.cal_compute_time(pub_edge, n)
        if pub_edge.used_volume <= pub_edge.max_volume:
            delay += gamma*self.cal_distance(pub_edge.x, pub_edge.y, sub_edge.x, sub_edge.y)
        else:
            delay += 2*self.cloud_time
        delay += gamma*self.cal_distance(sub_edge.x, sub_edge.y, subscriber.x, subscriber.y)

        return delay

    
    #  処理時間の計算
    def cal_compute_time(self, edge, n):
        topic = self.all_topic[n]

        if edge.used_volume <= edge.max_volume:
            delay = (topic.require_cycle*(topic.volume / topic.data_size)) / edge.power_allocation[n]
        else:
            delay = (topic.require_cycle*(topic.volume / topic.data_size)) / self.cloud_cycle

        return delay
    

    #  2点間の距離の計算
    def cal_distance(self, x1, y1, x2, y2):
        return ((int)(math.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))*100) / 100)
