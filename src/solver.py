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
        self.cpu_power_gain = 1/self.cpu_power
        self.power_allocation = np.zeros(num_topic)
        self.power_gain = np.zeros(num_topic)
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
        self.cloud_cycle_gain = 1/self.cloud_cycle

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

        """
        self.all_client = []
        num_publisher = np.zeros(self.num_topic)
        for _ in range(self.num_client):
            c = self.data_set.pop(0)
            client = Client(c.id, c.x, c.y, c.pub_topic, c.sub_topic, self.num_topic)
            self.all_client.append(client)

            for n in range(self.num_topic):
                if client.pub_topic[n] == 1:
                    num_publisher[n] += 1

        for topic in self.all_topic:
            topic.update_client(num_publisher[n], self.time_step)
            topic.cal_volume(self.time_step)
        """
    

    def update_client(self):
        self.all_client = []
        num_publisher = np.zeros(self.num_topic)
        for _ in range(self.num_client):
            c = self.data_set.pop(0)
            client = Client(c.id, c.x, c.y, c.pub_topic, c.sub_topic, self.num_topic)
            self.all_client.append(client)

            for n in range(self.num_topic):
                if client.pub_topic[n] == 1:
                    num_publisher[n] += 1

        for topic in self.all_topic:
            topic.update_client(num_publisher[n], self.time_step)
            topic.cal_volume(self.time_step)


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


    def solve(self, time, d, d_s, p, s, output_file):
        num_data = np.zeros(self.num_topic)
        for n in range(self.num_topic):
            topic = self.all_topic[n]
            num_data[n] = topic.volume/topic.data_size
        
        #  最適化問題の定式化
        model = grb.Model("model_" + str(time))

        #  変数の定式化
        x = {}
        for m in range(self.num_client):
            for n in p[m]:
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
            for n in p[m]:
                for m2 in s[n]:
                    for l in range(self.num_edge):
                        for l2 in range(self.num_edge):
                            w[m, n, m2, l, l2] = model.addVar(vtype=grb.GRB.BINARY, name=f"w({m}, {n}, {m2}, {l}, {l2})")
        
        v = {}
        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    v[m, n, l] = model.addVar(vtype=grb.GRB.BINARY, name=f"v({m}, {n}, {l})")
        
        num_user = {}
        for l in range(self.num_edge):
            num_user[l] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"num_user({l})")
        
        compute_time = {}
        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    compute_time[m, n, l] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"compute_fime({m}, {n}, {l})")
        
        model.update()

        #  制約式の定義
        for m in range(self.num_client):
            for n in p[m]:
                model.addConstr(grb.quicksum(x[m, n, l] for l in range(self.num_edge)) == 1, name=f"con_x({m}, {n})")

        for m in range(self.num_client):
            model.addConstr(grb.quicksum(y[m, l] for l in range(self.num_edge)) == 1, name="con_y({m})")

        for l in range(self.num_edge):
            model.addConstr(grb.quicksum(z[l, n]*self.all_topic[n].volume for n in range(self.num_topic)) <= self.all_edge[l].max_volume, name=f"con_z({l})")

        for m in range(self.num_client):
            for n in p[m]:
                for m2 in s[n]:
                    for l in range(self.num_edge):
                        for l2 in range(self.num_edge):
                            model.addConstr(w[m, n, m2, l, l2] == x[m, n, l]*y[m2, l2], name=f"con_w({m}, {n}, {m2}, {l}, {l2})")

        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    model.addConstr(v[m, n, l] == x[m, n, l]*z[l, n], name=f"con_v({m, n, l})")

        for l in range(self.num_edge):
            model.addConstr(num_user[l] == grb.quicksum(x[m ,n, l] for m in range(self.num_client) for n in p[m]), name=f"con_num_user({n}, {l})")

        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    model.addConstr(compute_time[m, n, l] == self.all_topic[n].require_cycle*num_data[n]*v[m, n, l]*num_user[l]*self.all_edge[l].cpu_power_gain + self.all_topic[n].require_cycle*num_data[n]*(1 - z[l, n])*x[m, n, l]*self.cloud_cycle_gain, name=f"compute_time({m}, {n}, {l})")

        model.update()

        #  目的関数の定義
        obj = grb.LinExpr()

        obj += grb.quicksum(grb.quicksum(d[m][l]*x[m, n, l] for l in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(compute_time[m, n, l] for l in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(2*self.cloud_time*(1-z[l, n])*x[m, n, l] for l in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(z[l, n]*d_s[l][l2]*w[m, n, m2, l ,l2] for l in range(self.num_edge) for l2 in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(d[m2][l2]*y[m2, l2] for l2 in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])

        model.setObjective(obj, sense=grb.GRB.MINIMIZE)

        model.update()

        # 求解
        model.Params.NonConvex = 2
        model.optimize()

        delay = self.output_solution(time, model, d, d_s, p, s, output_file)

        return delay

    
    def solve_y_fix(self, time, d, d_s, p, s, output_file):
        num_data = np.zeros(self.num_topic)
        for n in range(self.num_topic):
            topic = self.all_topic[n]
            num_data[n] = topic.volume/topic.data_size
        
        #  最適化問題の定式化
        model = grb.Model("model_" + str(time))

        #  変数の定式化
        x = {}
        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    x[m, n, l] = model.addVar(vtype=grb.GRB.BINARY, name=f"x({m}, {n}, {l})")

        y = np.zeros((self.num_client, self.num_edge))
        for n in range(self.num_topic):
            for m in s[n]:
                client = self.all_client[m]
                min_idx = -1
                min_dis = 1000000000
                for l in range(self.num_edge):
                    edge = self.all_edge[l]
                    distance = util.cal_distance(client.x, client.y, edge.x, edge.y)
                    if distance < min_dis:
                        min_idx = l
                        min_dis = distance
                
                y[m][min_idx] = 1

        z = {}
        for l in range(self.num_edge):
            for n in range(self.num_topic):
                z[l, n] = model.addVar(vtype=grb.GRB.BINARY, name=f"z({l}, {n})")

        w = {}
        for m in range(self.num_client):
            for n in p[m]:
                for m2 in s[n]:
                    for l in range(self.num_edge):
                        for l2 in range(self.num_edge):
                            w[m, n, m2, l, l2] = model.addVar(vtype=grb.GRB.BINARY, name=f"w({m}, {n}, {m2}, {l}, {l2})")
        
        v = {}
        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    v[m, n, l] = model.addVar(vtype=grb.GRB.BINARY, name=f"v({m}, {n}, {l})")
        
        num_user = {}
        for l in range(self.num_edge):
            num_user[l] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"num_user({l})")
        
        compute_time = {}
        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    compute_time[m, n, l] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"compute_fime({m}, {n}, {l})")
        
        model.update()

        #  制約式の定義
        for m in range(self.num_client):
            for n in p[m]:
                model.addConstr(grb.quicksum(x[m, n, l] for l in range(self.num_edge)) == 1, name=f"con_x({m}, {n})")

        for l in range(self.num_edge):
            model.addConstr(grb.quicksum(z[l, n]*self.all_topic[n].volume for n in range(self.num_topic)) <= self.all_edge[l].max_volume, name=f"con_z({l})")

        for m in range(self.num_client):
            for n in p[m]:
                for m2 in s[n]:
                    for l in range(self.num_edge):
                        for l2 in range(self.num_edge):
                            model.addConstr(w[m, n, m2, l, l2] == x[m, n, l]*y[m2, l2], name=f"con_w({m}, {n}, {m2}, {l}, {l2})")

        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    model.addConstr(v[m, n, l] == x[m, n, l]*z[l, n], name=f"con_v({m, n, l})")

        for l in range(self.num_edge):
            model.addConstr(num_user[l] == grb.quicksum(x[m ,n, l] for m in range(self.num_client) for n in p[m]), name=f"con_num_user({n}, {l})")

        for m in range(self.num_client):
            for n in p[m]:
                for l in range(self.num_edge):
                    model.addConstr(compute_time[m, n, l] == self.all_topic[n].require_cycle*num_data[n]*v[m, n, l]*num_user[l]*self.all_edge[l].cpu_power_gain + self.all_topic[n].require_cycle*num_data[n]*(1 - z[l, n])*x[m, n, l]*self.cloud_cycle_gain, name=f"compute_time({m}, {n}, {l})")

        model.update()

        #  目的関数の定義
        obj = grb.LinExpr()

        obj += grb.quicksum(grb.quicksum(d[m][l]*x[m, n, l] for l in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(compute_time[m, n, l] for l in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(2*self.cloud_time*(1-z[l, n])*x[m, n, l] for l in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(z[l, n]*d_s[l][l2]*w[m, n, m2, l ,l2] for l in range(self.num_edge) for l2 in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])
        obj += grb.quicksum(grb.quicksum(d[m2][l2]*y[m2, l2] for l2 in range(self.num_edge)) for m in range(self.num_client) for n in p[m] for m2 in s[n])

        model.setObjective(obj, sense=grb.GRB.MINIMIZE)

        model.update()

        # 求解
        model.Params.NonConvex = 2
        model.optimize()

        delay = self.output_solution_y_fix(time, model, d, d_s, p, s, y, output_file)

        return delay


    #  出力形式を決定し、追記する必要あり
    def output_solution(self, time, model, d, d_s, p, s, output_file):
        opt = []
        x_opt = np.zeros((self.num_client, self.num_topic, self.num_edge))
        y_opt = np.zeros((self.num_client, self.num_edge))
        z_opt = np.zeros((self.num_edge, self.num_topic))
        total_delay = 0.0

        if model.Status == grb.GRB.OPTIMAL:
            print(" 最適解 ")
            print(model.ObjVal)
            for v in model.getVars():
                #print(v.VarName, v.X)
                opt.append(v.X)
            
            # 最適解の取り出し
            for m in range(self.num_client):
                for n in p[m]:
                    for l in range(self.num_edge):
                        x_opt[m][n][l] = opt.pop(0)
            
            for m in range(self.num_client):
                for l in range(self.num_edge):
                    y_opt[m][l] = opt.pop(0)

            for l in range(self.num_edge):
                for n in range(self.num_topic):
                    z_opt[l][n] = opt.pop(0)
            
            for m in range(self.num_client):
                for n in p[m]:
                    for m2 in s[n]:
                        for l in range(self.num_edge):
                            for l2 in range(self.num_edge):
                                #  wの取り出し
                                opt.pop(0)
            
            v_opt = np.zeros((self.num_client, self.num_topic, self.num_edge))
            for m in range(self.num_client):
                for n in p[m]:
                    for l in range(self.num_edge):
                        v_opt[m][n][l] = opt.pop(0)
            
            num_user = np.zeros(self.num_edge)
            for l in range(self.num_edge):
                num_user[l] = opt.pop(0)
            
            compute_time_opt = np.zeros((self.num_client, self.num_topic, self.num_edge))
            for m in range(self.num_client):
                for n in p[m]:
                    for l in range(self.num_edge):
                        compute_time_opt[m][n][l] = opt.pop(0)
                        
            # 遅延を格納する変数
            delay = np.zeros((self.num_topic, self.num_client, self.num_client))
            num_user = np.zeros(self.num_edge)
            for l in range(self.num_edge):
                for m in range(self.num_client):
                    for n in p[m]:
                        num_user[l] += x_opt[m][n][l]

            x_opt_output = np.ones((self.num_client, self.num_topic))*-1
            y_opt_output = np.ones((self.num_client))*-1
            for m in range(self.num_client):
                for n in range(self.num_topic):
                    for l in range(self.num_edge):
                        if x_opt[m][n][l] == 1:
                            x_opt_output[m][n] = l
                
                for l in range(self.num_edge):
                        if y_opt[m][l] == 1:
                            y_opt_output[m] = self.all_edge[l].id
            
            with open(output_file, "a") as f:
                for m in range(self.num_client):
                    f.write(f"{m},{time}")

                    for n in range(self.num_topic):
                        f.write(f",{x_opt_output[m][n]}")

                    for n in range(self.num_topic):
                        if self.all_client[m].sub_topic[n] == 1:
                            f.write(f",{y_opt_output[m]}\n")
                        else:
                            f.write(",-1")

            # 総遅延の計算
            for m in range(self.num_client):
                for n in p[m]:
                    for m2 in s[n]:
                        delay[n][m][m2] = calmyModeltime(m, m2, n, x_opt, y_opt, z_opt, d, d_s, num_user, self.all_topic, self.all_edge, self.cloud_time, self.cloud_cycle)
                        total_delay += delay[n][m][m2]
            
            print("total delay = ")
            print(total_delay)

        else:
            print("実行不可能")

        return total_delay


#  出力形式を決定し、追記する必要あり
    def output_solution_y_fix(self, time, model, d, d_s, p, s, y_opt, output_file):
        opt = []
        x_opt = np.zeros((self.num_client, self.num_topic, self.num_edge))
        z_opt = np.zeros((self.num_edge, self.num_topic))
        total_delay = 0.0

        if model.Status == grb.GRB.OPTIMAL:
            print(" 最適解 ")
            print(model.ObjVal)
            for v in model.getVars():
                #print(v.VarName, v.X)
                opt.append(v.X)
            
            # 最適解の取り出し
            for m in range(self.num_client):
                for n in p[m]:
                    for l in range(self.num_edge):
                        x_opt[m][n][l] = opt.pop(0)

            for l in range(self.num_edge):
                for n in range(self.num_topic):
                    z_opt[l][n] = opt.pop(0)
            
            for m in range(self.num_client):
                for n in p[m]:
                    for m2 in s[n]:
                        for l in range(self.num_edge):
                            for l2 in range(self.num_edge):
                                #  wの取り出し
                                opt.pop(0)
            
            v_opt = np.zeros((self.num_client, self.num_topic, self.num_edge))
            for m in range(self.num_client):
                for n in p[m]:
                    for l in range(self.num_edge):
                        v_opt[m][n][l] = opt.pop(0)
            
            num_user = np.zeros(self.num_edge)
            for l in range(self.num_edge):
                num_user[l] = opt.pop(0)
            
            compute_time_opt = np.zeros((self.num_client, self.num_topic, self.num_edge))
            for m in range(self.num_client):
                for n in p[m]:
                    for l in range(self.num_edge):
                        compute_time_opt[m][n][l] = opt.pop(0)
                        
            # 遅延を格納する変数
            delay = np.zeros((self.num_topic, self.num_client, self.num_client))
            num_user = np.zeros(self.num_edge)
            for l in range(self.num_edge):
                for m in range(self.num_client):
                    for n in p[m]:
                        num_user[l] += x_opt[m][n][l]

            x_opt_output = np.ones((self.num_client, self.num_topic))*-1
            y_opt_output = np.ones((self.num_client))*-1
            for m in range(self.num_client):
                for n in range(self.num_topic):
                    for l in range(self.num_edge):
                        if x_opt[m][n][l] == 1:
                            x_opt_output[m][n] = l
                
                for l in range(self.num_edge):
                    if y_opt[m][l] == 1:
                        y_opt_output[m] = self.all_edge[l].id
            
            with open(output_file, "a") as f:
                for m in range(self.num_client):
                    f.write(f"{m},{time},{self.all_client[m].x},{self.all_client[m].y}")

                    for n in range(self.num_topic):
                        f.write(f",{x_opt_output[m][n]}")

                    for n in range(self.num_topic):
                        if self.all_client[m].sub_topic[n] == 1:
                            f.write(f",{y_opt_output[m]}")
                        else:
                            f.write(",-1")

                    f.write("\n")

            # 総遅延の計算
            for m in range(self.num_client):
                for n in p[m]:
                    for m2 in s[n]:
                        delay[n][m][m2] = calmyModeltime(m, m2, n, x_opt, y_opt, z_opt, d, d_s, num_user, self.all_topic, self.all_edge, self.cloud_time, self.cloud_cycle)
                        total_delay += delay[n][m][m2]
            
            print("total delay = ")
            print(total_delay)
        else:
            print("実行不可能")

        return total_delay


# 提案モデルの遅延の合計を計算
def calmyModeltime(m, m2, n, x, y, z, d, d_s, num_user, all_topic, all_edge, cloud_time, cloud_cycle):
    time = 0.0
    topic = all_topic[n]
    num_data = topic.volume/topic.data_size

    for l in range(len(all_edge)):
        time += d[m][l]*x[m][n][l]

    for l in range(len(all_edge)):
        if x[m][n][l] == 1:
            time_front = (x[m][n][l]*topic.require_cycle * num_data)
            time_back = (x[m][n][l]*z[l][n]*(num_user[l]/all_edge[l].cpu_power) + (1 - z[l][n])*x[m][n][l]/cloud_cycle)
            time += time_front*time_back

    for l in range(len(all_edge)):
        time += 2*cloud_time*(1-z[l][n])*x[m][n][l]

    for l in range(len(all_edge)):
        for l2 in range(len(all_edge)):
            time += z[l][n]*d_s[l][l2]*x[m][n][l]*y[m2][l2]

    for l2 in range(len(all_edge)):
        time += d[m2][l2]*y[m2][l2]

    return time
