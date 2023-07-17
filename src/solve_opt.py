from solver import Solver
import pandas as pd
import time as time_modu

if __name__ == '__main__':
    index_file = "../dataset/debug/index/index_hard.csv"
    output_file = "../dataset/debug/solution/solution_hard_opt.csv"

    start_time = 0

    solver = Solver(index_file)

    if start_time == 0:
        with open(output_file, "w") as f:
            pass

    total_delay = 0.0

    for time in range(0, start_time, solver.time_step):
        solver.update_client()

    start = time_modu.perf_counter()

    for time in range(start_time, solver.simulation_time, solver.time_step):
        print(f"time = {time}")
        solver.update_client()

        d, d_s = solver.set_delay()

        p, s = solver.set_pub_sub()
        
        print("\n----------start solve----------\n")
        delay = solver.solve_y_fix(time, d, d_s, p, s, output_file)
        total_delay += delay
        print("\n----------end solve----------\n")

    end = time_modu.perf_counter()

    print(f"total_delay = {total_delay}")

    print(f"compute time = {end - start}")

    df_index = pd.read_csv(index_file, index_col=0)
    df_index.at['data', 'opt'] = total_delay

    df_index.to_csv(index_file)