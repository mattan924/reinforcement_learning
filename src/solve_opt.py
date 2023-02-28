from solver import Solver

if __name__ == '__main__':
    index_file = "../dataset/learning_data/index/index_test.csv"
    output_file = "../dataset/learning_data/solution/solution_test.csv"

    solver = Solver(index_file)

    with open(output_file, "w") as f:
        f.write(f"id, time")

        for n in range(solver.num_topic):
            f.write(f",pub{n}")
        
        f.write(f",sub\n")

    total_delay = 0.0

    for time in range(0, solver.simulation_time, solver.time_step):
        print(f"time = {time}")
        solver.update_client()

        d, d_s = solver.set_delay()

        p, s = solver.set_pub_sub()
            
        print("\n----------start solve----------\n")
        delay = solver.solve_y_fix(time, d, d_s, p, s, output_file)
        total_delay += delay
        print("\n----------end solve----------\n")

    print(f"total_delay = {total_delay}")