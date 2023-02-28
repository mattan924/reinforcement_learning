from solver import Solver

if __name__ == '__main__':
    index_file = "../dataset/learning_data/index/index_test.csv"

    solver = Solver(index_file)

    for time in range(0, solver.simulation_time, solver.time_step):
        d, d_s = solver.set_delay()

        if time != 0:
            solver.update_client()

        p, s = solver.set_pub_sub()
            
        print("\n----------start solve----------\n")
        solver.solve(time, d, d_s, p, s)
        print("\n----------end solve----------\n")