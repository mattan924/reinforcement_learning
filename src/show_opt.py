import sys
sys.path.append("../../dataset_visualization/src/")
import animation


index_file = "../dataset/debug/debug/index/index_hard.csv"

opt_solution_file = "../dataset/debug/debug/solution/solution_hard_opt.csv"

output_file = "../dataset/debug/debug/animation/animation_hard_opt.gif"

animation.create_opt_animation(index_file, output_file, opt_solution_file, 10)