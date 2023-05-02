from solver import Solver
import pandas as pd

if __name__ == '__main__':
    index_file = "../dataset/learning_data/index/index_multi2.csv"
    output_file = "../dataset/learning_data/solution/solution_multi2.csv"

    start_time = 0

    solver = Solver(index_file)

    if start_time == 0:
        with open(output_file, "w") as f:
            pass

    total_delay = 0.0

    for time in range(0, start_time, solver.time_step):
        solver.update_client()

    for time in range(start_time, solver.simulation_time, solver.time_step):
        print(f"time = {time}")
        solver.update_client()

        d, d_s = solver.set_delay()

        p, s = solver.set_pub_sub()
            
        print("\n----------start solve----------\n")
        delay = solver.solve_y_fix(time, d, d_s, p, s, output_file)
        total_delay += delay
        print("\n----------end solve----------\n")

    print(f"total_delay = {total_delay}")

    df_index = pd.read_csv(index_file, index_col=0)
    df_index.at['data', 'opt'] = total_delay

    df_index.to_csv(index_file)