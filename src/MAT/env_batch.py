from ..env import Env
import numpy as np
import sys
sys.path.append("../../dataset_visualization/src")
import util
import pandas as pd


class Env_Batch:

    def __init__(self, index_file_list):
        self.batch_size = len(index_file_list)

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
        self.topic_total_num_client = np.zeros((self.batch_size, self.num_topic))
        self.topic_volume = np.zeros((self.batch_size, self.num_topic))

        self.client_x = np.zeros((self.batch_size, self.num_client))
        self.client_y = np.zeros((self.batch_size, self.num_client))
        self.client_pub_topic = np.zeros((self.batch_size, self.num_client, self.num_topic))
        self.client_sub_topic = np.zeros((self.batch_size, self.num_client, self.num_topic))
        self.client_pub_edge = np.zeros((self.batch_size, self.num_client, self.num_topic, self.num_edge))
        self.client_sub_edge = np.zeros((self.batch_size, self.num_client, self.num_edge))

        for batch_idx in range(self.batch_size):
            df_index = pd.read_csv(index_file_list[batch_idx], index_col=0)
            data_file = df_index.at['data', 'assign_file']
            edge_file = df_index.at['data', 'edge_file']
            topic_file = df_index.at['data', 'topic_file']

            self.init_edge = util.read_edge(edge_file)
            for edge_idx in range(self.num_edge):
                edge = self.init_edge[edge_idx]
                self.edge_x[batch_idx][edge_idx] = edge.x 
                self.edge_y[batch_idx][edge_idx] = edge.y
                self.edge_power_allocation[batch_idx][edge_idx] = edge.cpy_cycle
                self.edge_max_volume[batch_idx][edge_idx] = edge.volume
                self.edge_cpu_cycle[batch_idx][edge_idx] = edge.cpu_cycle

            self.init_topic = util.read_topic(topic_file)
            for topic_idx in range(self.num_topic):
                topic = self.init_topic[topic_idx]

                self.topic_save_period[batch_idx][topic_idx] = topic.save_period
                self.topic_publish_rate[batch_idx][topic_idx] = topic.publish_rate
                self.topic_data_size[batch_idx][topic_idx] = topic.data_size
                self.topic_require_cycle[batch_idx][topic_idx] = topic.require_cycle

            self.learning_data = util.read_data_set_topic(data_file, self.num_topic)
            for client_idx in range(self.num_client):
                client = self.learning_data.pop(0)

                self.client_x[batch_idx][client_idx] = client.x
                self.client_y[batch_idx][client_idx] = client.y
                self.client_pub_topic[batch_idx][client_idx] = client.pub_topic
                self.client_sub_topic[batch_idx][client_idx] = client.sub_topic


    def topic_update_client(self, time_step):
        self.topic_num_client_history[:, :, time_step] = self.client_pub_topic[:, ]


    def topic_cal_volume(self):
        pass