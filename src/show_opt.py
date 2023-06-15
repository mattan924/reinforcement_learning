import sys
sys.path.append("../../dataset_visualization/src/")
import animation


index_file = "../dataset/learning_data/index/index_multi2.csv"

opt_solution_file = "../dataset/learning_data/solution/solution_multi_opt2.csv"

output_file = "../dataset/learning_data/animation/animation_multi_opt2.gif"

animation.create_opt_animation(index_file, output_file, opt_solution_file, 10)