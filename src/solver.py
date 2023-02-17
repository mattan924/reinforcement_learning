import gurobipy as grb
import pandas as pd
import numpy as np
import queue
import sys
sys.path.append("../../dataset_visualization/src")
import util


class Client:

    def __init__(self, id, x, y, pub_topic, sub_topic, num_topic):
        self.id = id
        self.x = x
        self.y = y
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.pub_edge = np.zeros(num_topic)
        self.sub_edge = np.zeros(num_topic)


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


    def cal_volume(self, time_step):
        self.volume = self.data_size*self.publish_rate*time_step*self.total_num_client

    
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


class Solver:

    def __init__(self, index_file):
        df_index = pd.read_csv(index_file, index_col=0)
        self.config_file = df_index.at['data', 'config_file']
        self.data_file = df_index.at['data', 'assign_file']
        self.edge_file = df_index.at['data', 'edge_file']
        self.topic_file = df_index.at['data', 'topic_file']

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

        #  0.1 (ms/km)
        self.gamma = 0.1

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

        self.data_set = util.read_data_set_topic(self.data_file, self.num_topic)

        self.all_client = []
        for _ in range(self.num_client):
            c = self.data_set.pop(0)
            client = Client(c.id, c.x, c.y, c.pub_topic, c.sub_topic, self.num_topic)
            self.all_client.append(client)
    

    def update_client(self):
        self.all_client = []
        for _ in range(self.num_client):
            c = self.data_set.pop(0)
            client = Client(c.id, c.x, c.y, c.pub_topic, c.sub_topic, self.num_topic)
            self.all_client.append(client)


    def set_delay(self):
        #  d: クライアントとエッジ間の遅延時間
        d = np.zeros((self.num_client, self.num_edge))

        #  d_s: エッジ間の遅延時間
        d_s = np.zeros((self.num_edge, self.num_edge))

        for client in self.all_client:
            for edge in self.all_edge:
                distance = util.cal_distance(client.x, client.y, edge.x, edge.y)
                distance = (int)(distance*100)
                d[client.id][edge.id] = (distance/100)*self.gamma
        
        for edge1 in self.all_edge:
            for edge2 in self.all_edge:
                distance = util.cal_distance(edge1.x, edge1.y, edge2.x, edge2.y)
                distance = (int)(distance*100)
                d_s[edge1.id][edge2.id] = (distance/100)*self.gamma

        return d, d_s
    

    def set_pub_sub(self):
        p = [[] for _ in range(self.num_client)]
        for m in range(self.num_client):
            for n in range(self.num_topic):
                if self.all_client[m].pub_topic[n] == 1:
                    p[m].append(n)
        
        s = [[] for _ in range(self.num_topic)]
        for m in range(self.num_client):
            for n in range(self.num_topic):
                if self.all_client[m].sub_topic[n] == 1:
                    s[n].append(m)
        
        return p, s


    def solve(self, d, d_s, p, s):
        #  最適化問題の定式化
        model = grb.Model("model")

        #  変数の定式化
        x = {}
        for m in range(self.num_client):
            for n in self.p[m]:
                for l in range(self.num_edge):
                    x[m, n, l] = model.addVar(vtype=grb.GRB.BINARY, name=f"x({m}, {n}, {l})")

        y = {}
        for m in range(self.num_client):
            for l in range(self.num_edge):
                y[m, l] = model.addVar(vtype=grb.GRB.BINARY, name=f"y({m},{l})")

        z = {}
        for l in range(self.num_edge):
            for n in range(self.num_topic):
                z[l, n] = model.addVar(vtype=grb.GRB.BINARY, name=f"z({l}, {n})")

        w = {}
        for m in range(self.num_client):
            for n in self.p[m]:
                for m2 in self.s[n]:
                    for l in range(self.num_edge):
                        for l2 in range(self.num_edge):
                            w[m, n, m2, l, l2] = model.addVar(vtype=grb.GRB.BINARY, name=f"w({m}, {n}, {m2}, {l}, {l2})")
        
        model.update()

        #  制約式の定義
        for m in range(self.num_client):
            for n in self.p[m]:
                model.addConstr(grb.quicksum(x[m, n, l] for l in range(self.num_edge)) == 1, name=f"con_x({m}, {n})")

        for m in range(self.num_client):
            model.addConstr(grb.quicksum(y[m, l] for l in range(self.num_edge)) == 1, name="con_y({m})")

        for l in range(self.num_edge):
            model.addConstr(grb.quicksum(z[l, n]*self.all_topic[n].volume for n in range(self.num_topic)) <= self.all_edge[l].max_volume, name=f"con_z({l})")

        for m in range(self.num_client):
            for n in self.p[m]:
                for m2 in self.s[n]:
                    for l in range(self.num_edge):
                        for l2 in range(self.num_edge):
                            model.addConstr(w[m, n, m2, l, l2] == x[m, n, l]*y[m2, l2], name=f"con_w({m}, {n}, {m2}, {l}, {l2})")

        model.update()

        num_data = np.zeros(self.num_topic)
        for n in range(self.num_topic):
            topic = self.all_topic[n]
            num_data[n] = topic.volume/topic.data_size

        #  目的関数の定義
        obj = grb.LinExpr()

        obj += grb.quicksum(grb.quicksum(self.d[m][l]*x[m, n, l] for l in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        #  修正及び、間違いがないかのチェックが必要
        obj += grb.quicksum((self.all_topic[n].require_cycle*num_data[n])/(grb.quicksum(x[m, n, l]*z[l, n]*self.all_edge[l].power_allocation[n] for l in range(self.num_edge)) + (1 - grb.quicksum(x[m, n, l]*z[l, n] for l in range(self.num_edge)))*self.cloud_cycle) for m in range(self.num_client) for n in self.p[m] for m2 in self.s[n])
        obj += grb.quicksum(grb.quicksum(2*self.cloud_time*(1-z[l, n])*x[m, n, l] for l in range(num_edge)) for m in range(num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(z[l, n]*d_s[l][l2]*w[m, n, m2, l ,l2] for l in range(num_edge) for l2 in range(num_edge)) for m in range(num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(d[m2][l2]*y[m2, l2] for l2 in range(num_edge)) for m in range(num_client) for n in p[m] for m2 in s[n])

        model.setObjective(obj, sense=grb.GRB.MINIMIZE)

        model.update()

        # 求解
        model.optimize()

        self.output_solution(model, d, d_s, p, s)


    #  出力形式を決定し、追記する必要あり
    def output_solution(self, model, d, d_s, p, s):
        v_opt = []
        x_opt = np.zeros((self.num_client, self.num_topic, self.num_edge))
        y_opt = np.zeros((self.num_client, self.num_edge))
        z_opt = np.zeros((self.num_edge, self.num_topic))

        if model.Status == grb.GRB.OPTIMAL:
            print(" 最適解 ")
            print(model.ObjVal)
            for v in model.getVars():
                #print(v.VarName, v.X)
                v_opt.append(v.X)
            
            # 最適解の取り出し
            for m in range(self.num_client):
                for n in p[m]:
                    for l in range(self.num_edge):
                        x_opt[m][n][l] = v_opt.pop(0)

            for m in range(self.num_client):
                for l in range(self.num_edge):
                    y_opt[m][l] = v_opt.pop(0)

            for l in range(self.num_edge):
                for n in range(self.num_topic):
                    z_opt[l][n] = v_opt.pop(0)

            # 遅延を格納する変数
            delay = np.zeros((self.num_topic, self.num_client, self.num_client))

            # 平均遅延の計算
            ave_time = 0.0
            cnt = 0
            for m in range(self.num_client):
                for n in p[m]:
                    for m2 in s[n]:
                        delay[n][m][m2] = calmyModeltime(m, m2, n, x_opt, y_opt, z_opt, d, d_s, self.all_topic, self.all_edge, self.cloud_time, self.cloud_cycle)
                        ave_time += delay[n][m][m2]
                        cnt += 1
            
            ave_time = ave_time/cnt
            print("average time = ")
            print(ave_time)
        else:
            print("実行不可能")


# 提案モデルの遅延の合計を計算
def calmyModeltime(m, m2, n, x, y, z, d, d_s, all_topic, all_edge, cloud_time, cloud_cycle):
    time = 0.0
    topic = all_topic[n]
    num_data = topic.volume/topic.data_size

    for l in range(len(all_edge)):
        time += d[m][l]*x[m][n][l]

    for l in range(len(all_edge)):
        time += (x[m][n][l]*topic.require_cycle * num_data) / (x[m][n][l]*z[l][n]*all_edge[l].power_allocation[n] + (1- z[l][n])*x[m][n][l]*cloud_cycle)

    for l in range(len(all_edge)):
        time += 2*cloud_time*(1-z[l][n])*x[m][n][l]

    for l in range(len(all_edge)):
        for l2 in range(len(all_edge)):
            time += z[l][n]*d_s[l][l2]*x[m][n][l]*y[m2][l2]

    for l2 in range(len(all_edge)):
        time += d[m2][l2]*y[m2][l2]

    return time
