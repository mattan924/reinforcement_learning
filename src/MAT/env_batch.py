import numpy as np
import sys
sys.path.append("../../dataset_visualization/src")
import util
import pandas as pd
import time as time_module


class Env_Batch:

    def __init__(self, index_file_list):
        self.batch_size = len(index_file_list)
        self.index_file_list = index_file_list

        df_index = pd.read_csv(index_file_list[0], index_col=0)
        self.config_file = df_index.at['data', 'config_file']
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

        self.client_x = np.zeros((self.batch_size, self.num_client))
        self.client_y = np.zeros((self.batch_size, self.num_client))
        self.client_pub_topic = np.zeros((self.batch_size, self.num_client, self.num_topic))
        self.client_sub_topic = np.zeros((self.batch_size, self.num_client, self.num_topic))
        self.client_pub_edge = np.zeros((self.batch_size, self.num_client, self.num_topic, self.num_edge))
        self.client_sub_edge = np.zeros((self.batch_size, self.num_client, self.num_edge))

        self.edge_x = np.zeros((self.batch_size, self.num_edge))
        self.edge_y = np.zeros((self.batch_size, self.num_edge))
        self.edge_max_volume = np.zeros((self.batch_size, self.num_edge))
        self.edge_used_volume = np.zeros((self.batch_size, self.num_edge, self.num_topic))
        self.edge_deploy_topic = np.bool_(np.zeros((self.batch_size, self.num_edge, self.num_topic)))
        self.edge_cpu_cycle = np.zeros((self.batch_size, self.num_edge))
        self.edge_power_allocation = np.zeros((self.batch_size, self.num_edge))
        self.edge_used_publisher = np.zeros((self.batch_size, self.num_edge, self.num_topic))
        self.edge_remain_cycle = np.zeros((self.batch_size, self.num_edge))

        self.topic_save_period = np.zeros((self.batch_size, self.num_topic))
        self.topic_publish_rate = np.zeros((self.batch_size, self.num_topic))
        self.topic_data_size = np.zeros((self.batch_size, self.num_topic))
        self.topic_require_cycle = np.zeros((self.batch_size, self.num_topic))
        self.topic_num_client_history = np.zeros((self.batch_size, self.num_topic, int(self.simulation_time / self.time_step)))
        self.topic_volume = np.zeros((self.batch_size, self.num_topic))

        self.init_edge_list = []
        self.init_topic_list = []
        self.learning_data_list = []

        for batch_idx in range(self.batch_size):
            df_index = pd.read_csv(self.index_file_list[batch_idx], index_col=0)
            data_file = df_index.at['data', 'assign_file']
            edge_file = df_index.at['data', 'edge_file']
            topic_file = df_index.at['data', 'topic_file']

            init_edge = util.read_edge(edge_file)
            self.init_edge_list.append(init_edge)
            for edge_idx in range(self.num_edge):
                edge = init_edge[edge_idx]
                self.edge_x[batch_idx][edge_idx] = edge.x 
                self.edge_y[batch_idx][edge_idx] = edge.y
                self.edge_power_allocation[batch_idx][edge_idx] = edge.cpu_cycle
                self.edge_max_volume[batch_idx][edge_idx] = edge.volume
                self.edge_cpu_cycle[batch_idx][edge_idx] = edge.cpu_cycle

            init_topic = util.read_topic(topic_file)
            self.init_topic_list.append(init_topic)
            for topic_idx in range(self.num_topic):
                topic = init_topic[topic_idx]

                self.topic_save_period[batch_idx][topic_idx] = topic.save_period
                self.topic_publish_rate[batch_idx][topic_idx] = topic.publish_rate
                self.topic_data_size[batch_idx][topic_idx] = topic.data_size
                self.topic_require_cycle[batch_idx][topic_idx] = topic.require_cycle

            learning_data = util.read_data_set_topic(data_file, self.num_topic)
            self.learning_data_list.append(learning_data)
            for client_idx in range(self.num_client):
                client = learning_data.pop(0)

                self.client_x[batch_idx][client_idx] = client.x
                self.client_y[batch_idx][client_idx] = client.y
                self.client_pub_topic[batch_idx][client_idx] = client.pub_topic
                self.client_sub_topic[batch_idx][client_idx] = client.sub_topic

        self.topic_update_client(time_step=0)

        self.topic_cal_volume()


    def topic_update_client(self, time_step):
        self.topic_num_client_history[:, :, time_step] = np.sum(self.client_pub_topic, axis=1)

        save_period = self.topic_save_period[0][0]
        cnt = np.sum(self.topic_save_period != save_period)

        if cnt == 0:
            start_idx = max(time_step - (save_period / self.time_step), 0)

            self.topic_total_num_client = np.sum(self.topic_num_client_history[:, :, start_idx:time_step+1], axis=2)            
        else:
            sys.exit("save_period が揃っていません")


    def topic_cal_volume(self):
        self.topic_volume = self.topic_data_size * self.topic_publish_rate * self.time_step * self.topic_total_num_client

    
    def reset(self):
        self.edge_used_publisher = np.zeros((self.batch_size, self.num_edge, self.num_topic))
        self.edge_used_volume = np.zeros((self.batch_size, self.num_edge, self.num_topic))
        self.edge_power_allocation = np.zeros((self.batch_size, self.num_edge))
        self.edge_deploy_topic = np.bool_((self.batch_size, self.num_edge, self.num_topic))

        self.topic_num_client_history = np.zeros((self.batch_size, self.num_topic, int(self.simulation_time / self.time_step)))

        self.client_x = np.zeros((self.batch_size, self.num_client))
        self.client_y = np.zeros((self.batch_size, self.num_client))
        self.client_pub_topic = np.zeros((self.batch_size, self.num_client, self.num_topic))
        self.client_sub_topic = np.zeros((self.batch_size, self.num_client, self.num_topic))
        self.client_pub_edge = np.zeros((self.batch_size, self.num_client, self.num_topic, self.num_edge))
        self.client_sub_edge = np.zeros((self.batch_size, self.num_client, self.num_edge))

        self.learning_data_list = []
        for batch_idx in range(self.batch_size):
            df_index = pd.read_csv(self.index_file_list[batch_idx], index_col=0)
            data_file = df_index.at['data', 'assign_file']

            for edge_idx in range(self.num_edge):
                edge = self.init_edge_list[batch_idx][edge_idx]
                self.edge_power_allocation[batch_idx][edge_idx] = edge.cpu_cycle

            learning_data = util.read_data_set_topic(data_file, self.num_topic)
            self.learning_data_list.append(learning_data)
            for client_idx in range(self.num_client):
                client = learning_data.pop(0)

                self.client_x[batch_idx][client_idx] = client.x
                self.client_y[batch_idx][client_idx] = client.y
                self.client_pub_topic[batch_idx][client_idx] = client.pub_topic
                self.client_sub_topic[batch_idx][client_idx] = client.sub_topic

        self.topic_update_client(0)

        self.topic_cal_volume()

    
    def get_perm(self, max_agent, max_topic):
        agent_perm_batch = np.zeros((self.batch_size, max_agent), dtype=np.int64)
        topic_perm_batch = np.zeros((self.batch_size, max_topic), dtype=np.int64)

        for idx in range(self.batch_size):
            agent_perm_batch[idx] = np.random.permutation(max_agent)
            topic_perm_batch[idx] = np.random.permutation(max_topic)

        return agent_perm_batch, topic_perm_batch

    
    def get_observation_mat(self, agent_perm_batch, topic_perm_batch, obs_size=9):
        channel_dim = obs_size * obs_size

        edge_obs_size = 3
        edge_channel_dim = 9
        topic_channel_dim = 3

        max_agent = agent_perm_batch.shape[1]
        max_topic = topic_perm_batch.shape[1]

        block_len_x = (self.max_x-self.min_x)/obs_size
        block_len_y = (self.max_y-self.min_y)/obs_size

        edge_block_len_x = (self.max_x-self.min_x)/edge_obs_size
        edge_block_len_y = (self.max_y-self.min_y)/edge_obs_size
        
        # 観測値
        obs_posi = np.zeros((self.batch_size, max_agent, channel_dim))  #  クライアントの位置
        obs_publisher = np.zeros((self.batch_size, max_topic, channel_dim))  #  あるトピックの publisher の分布
        obs_subscriber = np.zeros((self.batch_size, max_topic ,channel_dim))  #  あるトピックの subscriber の分布
        obs_distribution = np.zeros((self.batch_size, channel_dim))  #  クライアントの分布
        obs_storage = np.zeros((self.batch_size, edge_channel_dim))  #  最大ストレージサイズ
        obs_cpu_cycle = np.zeros((self.batch_size, edge_channel_dim))  #  CPU の最大クロック数
        obs_remain_cycle = np.zeros((self.batch_size, edge_channel_dim)) # 残りの計算負荷
        obs_topic_info = np.zeros((self.batch_size, max_topic, topic_channel_dim))  #  あるトピックの処理に必要なクロック数, データサイズ, ストレージサイズ

        mask = np.zeros((self.batch_size, max_agent, max_topic), dtype=np.bool)

        agent_id_mask = agent_perm_batch < self.num_client
        topic_id_mask = topic_perm_batch < self.num_topic

        # クライアントの位置を求める
        block_index_x_batch = np.clip(self.client_x / block_len_x, 0, obs_size-1).astype(int)
        block_index_y_batch = np.clip(self.client_y / block_len_y, 0, obs_size-1).astype(int)

        block_index = block_index_y_batch*obs_size + block_index_x_batch
        # block_index.shape is (batch_size, num_client)

        # クライアントの位置の情報を agent_perm に従って並び変える
        agent_perm_mask = agent_perm_batch[agent_id_mask].reshape(self.batch_size, self.num_client)
        topic_perm_mask = topic_perm_batch[topic_id_mask].reshape(self.batch_size, self.num_topic)

        block_index_perm = np.array([block_index[idx][agent_perm_mask[idx]] for idx in range(self.batch_size)])
        # block_index_perm.shape is (batch_size, num_client)

        block_index_perm_onehot = np.identity(channel_dim)[block_index_perm]
        # block_index_perm_onehot.shape is (batch_size, num_client, channel_dim)


        # obs_posi
        obs_posi_mask = np.zeros((self.batch_size, max_agent, channel_dim), dtype=np.bool_)
        obs_posi_mask[agent_id_mask] = block_index_perm_onehot.reshape(-1, channel_dim)
        
        obs_posi[obs_posi_mask] = 1000

        # obs_publisher
        # agent_perm の順に self.client_pub_topic を並び替え
        client_pub_topic_perm = np.array([self.client_pub_topic[idx][agent_perm_mask[idx]] for idx in range(self.batch_size)], dtype=np.bool_)

        # block_index_perm_onehot (batch_size, num_client, channel_dim) → (batch_size, num_client, num_topic, channel_dim) へ拡張
        out1 = np.repeat(block_index_perm_onehot[:, :, None, :], self.num_topic, axis=2)
        # client_pub_topic_perm (batch_size, num_client, num_topic) →  (batch_size, num_client, num_topic, channel_dim) へ拡張
        out2 = np.repeat(client_pub_topic_perm[:, :, :, None], channel_dim, axis=3)
        out3 = np.sum(out1 * out2, axis=1)

        obs_publisher[topic_id_mask] = np.array([out3[idx][topic_perm_mask[idx]] for idx in range(self.batch_size)]).reshape(-1, channel_dim)

        # obs_subscriber
        client_sub_topic_perm = np.array([self.client_sub_topic[idx][agent_perm_mask[idx]] for idx in range(self.batch_size)], dtype=np.bool_)

        # client_pub_topic_perm (batch_size, num_client, num_topic) →  (batch_size, num_client, num_topic, channel_dim) へ拡張
        out2 = np.repeat(client_sub_topic_perm[:, :, :, None], channel_dim, axis=3)
        out3 = np.sum(out1 * out2, axis=1)

        obs_subscriber[topic_id_mask] = np.array([out3[idx][topic_perm_mask[idx]] for idx in range(self.batch_size)]).reshape(-1, channel_dim)

        # obs_distribution
        obs_distribution = np.sum(block_index_perm_onehot, axis=1)

        # obs_storage

        # エッジの位置を求める
        block_index_edge_x_batch = np.clip(self.edge_x / edge_block_len_x, 0, edge_obs_size-1).astype(int)
        block_index_edge_y_batch = np.clip(self.edge_y / edge_block_len_y, 0, edge_obs_size-1).astype(int)

        block_index_edge = block_index_edge_y_batch*edge_obs_size + block_index_edge_x_batch

        block_index_edge_onehot = np.array(np.identity(edge_channel_dim)[block_index_edge], dtype=np.bool_)
        
        obs_storage = np.repeat(self.edge_max_volume[:, :, None], edge_channel_dim, axis=2)[block_index_edge_onehot].reshape(self.batch_size, edge_channel_dim)

        # obs_cpu_cycle
        obs_cpu_cycle = np.repeat(self.edge_cpu_cycle[:, :, None], edge_channel_dim, axis=2)[block_index_edge_onehot].reshape(self.batch_size, edge_channel_dim)

        # obs_remain_cycle
        obs_remain_cycle = np.repeat(self.edge_remain_cycle[:, :, None], edge_channel_dim, axis=2)[block_index_edge_onehot].reshape(self.batch_size, edge_channel_dim)

        # obs_topic_info
        a1 = np.array([self.topic_require_cycle[idx][topic_perm_mask[idx]].reshape(-1) * 1e4 for idx in range(self.batch_size)])
        a2 = np.array([self.topic_data_size[idx][topic_perm_mask[idx]].reshape(-1) * 1e4 for idx in range(self.batch_size)])
        a3 = np.array([self.topic_volume[idx][topic_perm_mask[idx]].reshape(-1) * 1e1 for idx in range(self.batch_size)])

        obs_topic_info[topic_id_mask] = np.stack([a1, a2, a3], axis=2).reshape(-1, topic_channel_dim)
        
        # obs_mask
        dummy = np.zeros((self.batch_size, self.num_client, int(max_topic - self.num_topic)))
        client_pub_topic_perm_extend = np.concatenate([client_pub_topic_perm, dummy], axis=2)

        client_pub_topic_perm_extend = np.array([client_pub_topic_perm_extend[idx][:, topic_perm_batch[idx].reshape(-1)] for idx in range(self.batch_size)])

        mask[agent_id_mask]= client_pub_topic_perm_extend.reshape(-1, max_topic)

        return obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_remain_cycle, obs_topic_info, mask
