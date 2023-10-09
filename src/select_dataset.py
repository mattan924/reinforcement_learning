import pandas
from env import Env


dataset_dir = "../dataset/data_set_hard/train/"

max_data_size = 10000

rate = 0.5

threshold = int(max_data_size * rate)

low_reward_data = 0
hight_reward_data = 0

while(1):
    if low_reward_data == threshold and hight_reward_data == threshold:
        break
    else:
        