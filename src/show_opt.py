import sys
sys.path.append("../../dataset_visualization/src")
import util
import animation


index_file = "../dataset/learning_data/index/index_multi.csv"

opt_solution_file = "../dataset/learning_data/solution/solution_multi.csv"

output_file = "../dataset/execution_data/animation/multi_opt.gif"

animation.create_opt_animation(index_file, output_file, opt_solution_file, 20)