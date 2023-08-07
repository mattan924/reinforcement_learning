import sys
sys.path.append("../../dataset_visualization/src/")
import animation


index_file = "../dataset/debug/index/index_onetopic.csv"

opt_solution_file = "../dataset/debug/solution/solution_onetopic_opt.csv"

output_file = "../dataset/debug/animation/animation_onetopic_opt.gif"

animation.create_opt_animation_onetopic(index_file, output_file, opt_solution_file, 10)