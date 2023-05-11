import sys
sys.path.append("../../dataset_visualization/src/")
import animation


index_file = "../dataset/data_set_hard/index/index_0.csv"

opt_solution_file = "../dataset/data_set_hard/solution/solution_0.csv"

output_file = "../dataset/data_set_hard/animation/animation_0_opt.gif"

animation.create_opt_animation(index_file, output_file, opt_solution_file, 20)