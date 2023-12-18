import numpy as np
import sys
sys.path.append("../../dataset_visualization/src")
import util
import pandas as pd


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

        self.edge_x = np.zeros((self.batch_size, self.num_edge))
        self.edge_y = np.zeros((self.batch_size, self.num_edge))
        self.edge_used_publisher = np.zeros((self.batch_size, self.num_edge, self.num_topic))
        self.edge_used_volume = np.zeros((self.batch_size, self.num_edge, self.num_topic))
        self.edge_power_allocation = np.zeros((self.batch_size, self.num_edge))
        self.edge_deploy_topic = np.bool_((self.batch_size, self.num_edge, self.num_topic))
        self.edge_max_volume = np.zeros((self.batch_size, self.num_edge))
        self.edge_cpu_cycle = np.zeros((self.batch_size, self.num_edge))

        self.topic_save_period = np.zeros((self.batch_size, self.num_topic))
        self.topic_publish_rate = np.zeros((self.batch_size, self.num_topic))
        self.topic_data_size = np.zeros((self.batch_size, self.num_topic))
        self.topic_require_cycle = np.zeros((self.batch_size, self.num_topic))
        self.topic_num_client_history = np.zeros((self.batch_size, self.num_topic, int(self.simulation_time / self.time_step)))

        self.client_x = np.zeros((self.batch_size, self.num_client))
        self.client_y = np.zeros((self.batch_size, self.num_client))
        self.client_pub_topic = np.zeros((self.batch_size, self.num_client, self.num_topic))
        self.client_sub_topic = np.zeros((self.batch_size, self.num_client, self.num_topic))
        self.client_pub_edge = np.zeros((self.batch_size, self.num_client, self.num_topic, self.num_edge))
        self.client_sub_edge = np.zeros((self.batch_size, self.num_client, self.num_edge))

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

        self.topic_update_client(0)

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

    
    def get_observation_mat(self, agent_perm_batch, topic_perm_batch, obs_size=81):
        channel_dim = obs_size * obs_size

        max_agent = agent_perm_batch.shape[1]
        max_topic = topic_perm_batch.shape[1]

        block_len_x = (self.max_x-self.min_x)/obs_size
        block_len_y = (self.max_y-self.min_y)/obs_size

        # old
        """
        obs_posi = np.zeros((self.batch_size, max_agent, channel_dim))  #  クライアントの位置
        obs_publisher = np.zeros((self.batch_size, max_topic, channel_dim))  #  あるトピックの publisher の分布
        obs_subscriber = np.zeros((self.batch_size, max_topic ,channel_dim))  #  あるトピックの subscriber の分布
        obs_distribution = np.zeros((self.batch_size, channel_dim))  #  クライアントの分布
        obs_storage = np.zeros((self.batch_size, channel_dim))  #  最大ストレージサイズ
        obs_cpu_cycle = np.zeros((self.batch_size, channel_dim))  #  CPU の最大クロック数
        obs_topic_info = np.zeros((self.batch_size, max_topic, 3))  #  あるトピックの処理に必要なクロック数, データサイズ, ストレージサイズ

        mask = np.zeros((self.batch_size, max_agent, max_topic))
        
        for batch_idx in range(self.batch_size):
            for i in range(max_agent):
                client_id = agent_perm_batch[batch_idx][i]

                if client_id < self.num_client:
                    block_index_x = np.clip(int(self.client_x[batch_idx][client_id] / block_len_x), 0, obs_size-1)
                    block_index_y = np.clip(int(self.client_y[batch_idx][client_id] / block_len_y), 0, obs_size-1)

                    obs_posi[batch_idx, i, block_index_y*obs_size + block_index_x] = 1000

                    for t in range(max_topic):
                        topic_id = topic_perm_batch[batch_idx][t]
                        
                        if topic_id < self.num_topic:
                            if self.client_pub_topic[batch_idx][client_id][topic_id] == 1:
                                obs_publisher[batch_idx][t][block_index_y*obs_size + block_index_x] += 1
                                mask[batch_idx][i][t] = 1

                            if self.client_sub_topic[batch_idx][client_id][topic_id] == 1:
                                obs_subscriber[batch_idx][t][block_index_y*obs_size + block_index_x] += 1

                    obs_distribution[batch_idx][block_index_y*obs_size + block_index_x] += 1

            for edge_id in range(self.num_edge):
                block_index_x = np.clip(int(self.edge_x[batch_idx][edge_id] / block_len_x), 0, obs_size-1)
                block_index_y = np.clip(int(self.edge_y[batch_idx][edge_id] / block_len_y), 0, obs_size-1)
                    
                obs_storage[batch_idx][block_index_y*obs_size + block_index_x] = self.edge_max_volume[batch_idx][edge_id]
                obs_cpu_cycle[batch_idx][block_index_y*obs_size + block_index_x] = self.edge_cpu_cycle[batch_idx][edge_id]

            for t in range(self.num_topic):
                topic_id = topic_perm_batch[batch_idx][t]
                if topic_id < self.num_topic:
                    obs_topic_info[t][0] = self.topic_require_cycle[batch_idx][topic_id] * 1e4
                    obs_topic_info[t][1] = self.topic_data_size[batch_idx][topic_id]* 1e4
                    obs_topic_info[t][2] = self.topic_volume[batch_idx][topic_id] * 1e1
        """
        
        # opt
        opt_obs_posi = np.zeros((self.batch_size, max_agent, channel_dim))  #  クライアントの位置
        opt_obs_publisher = np.zeros((self.batch_size, max_topic, channel_dim))  #  あるトピックの publisher の分布
        opt_obs_subscriber = np.zeros((self.batch_size, max_topic ,channel_dim))  #  あるトピックの subscriber の分布
        opt_obs_distribution = np.zeros((self.batch_size, channel_dim))  #  クライアントの分布
        opt_obs_storage = np.zeros((self.batch_size, channel_dim))  #  最大ストレージサイズ
        opt_obs_cpu_cycle = np.zeros((self.batch_size, channel_dim))  #  CPU の最大クロック数
        opt_obs_topic_info = np.zeros((self.batch_size, max_topic, 3))  #  あるトピックの処理に必要なクロック数, データサイズ, ストレージサイズ

        opt_mask = np.zeros((self.batch_size, max_agent, max_topic))

        # agent_perm_batch.shape = (15, 30)
        # topic_perm_batch.shape = (15, 3)

        opt_block_index_x_batch = np.clip(self.client_x / block_len_x, 0, obs_size-1).astype(int)
        opt_block_index_y_batch = np.clip(self.client_y / block_len_y, 0, obs_size-1).astype(int)
        # opt_block_index_x_batch.shape = (15, 15)
        # opt_block_index_y_batch.shape = (15, 15)

        opt_client_id_batch_mask = (agent_perm_batch < self.num_client)
        # opt_client_id_batch_mask.shape = (15, 30)

        opt_client_id_batch = agent_perm_batch[opt_client_id_batch_mask].reshape(self.batch_size, self.num_client)
        # opt_client_id_batch.shape = (15, 15)

        print(f"opt_client_id_batch = {opt_client_id_batch}")
        print(f"client_x = {self.client_x}")
        print(f"client_x_batch = {self.client_x[opt_client_id_batch].shape}")

        tmp_list = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        indices = [[0, 1], [1, 0], [2, 2]]

        print(f"tmp_list = {tmp_list}")



        return obs_posi, obs_publisher, obs_subscriber, obs_distribution, obs_storage, obs_cpu_cycle, obs_topic_info, mask
