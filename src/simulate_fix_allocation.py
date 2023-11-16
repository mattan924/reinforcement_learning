from solver import Solver
import pandas as pd

index_file = "../../reinforcement_learning/dataset/debug/debug/index/index_hard_hight_load.csv"
output_file = "../../reinforcement_learning/dataset/debug/debug/solution/solution_hard_hight_load_opt.csv"

start_time = 0

solver = Solver(index_file)

solver.update_client()

d, d_s = solver.set_delay()

p, s = solver.set_pub_sub()

delay = solver.solve_y_fix(start_time, d, d_s, p, s, output_file)

