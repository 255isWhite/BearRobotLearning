import json
import os
import numpy as np
from tqdm import tqdm

prefix = "data/"
base_dir = "libero/dataset/data_jpg/"

task_suite_name = 'libero_goal'
base_dir = os.path.join(prefix, base_dir)

dataset_lists = os.listdir(base_dir)

data = []
count = 0
for dataset in tqdm(dataset_lists):
    print(dataset)
    if dataset != task_suite_name:
        continue
    dataset_dir = os.path.join(base_dir, dataset)
    task_lists = os.listdir(dataset_dir)
    
    for task in task_lists:
        task_dir = os.path.join(dataset_dir, task)
        task_dir = task_dir + '/data'
        demo_lists = os.listdir(task_dir)
        
        instruction = task
        for demo in demo_lists:
            print("demo ",demo)
            demo_dir = os.path.join(task_dir, demo)
            
            action = np.load(demo_dir + f"/actions/data_{demo}_actions.npy")
            state = np.load(demo_dir + f"/states/data_{demo}_states.npy")
            
            length = action.shape[0]
            for i in range(length):
                save_dict = {}
                save_dict["instruction"] = instruction
                save_dict['D435_image'] = (demo_dir + f"/obs/agentview_rgb/data_{demo}_obs_agentview_rgb/agentview_rgb_{i}.jpg")[5:]
                save_dict['wrist_image'] = (demo_dir + f"/obs/eye_in_hand_rgb/data_{demo}_obs_eye_in_hand_rgb/eye_in_hand_rgb_{i}.jpg")[5:]
                save_dict["state"] = state[i].tolist()
                
                # action chunking a
                action_list = []
                for idx in range(6):
                    a = action[min(i+idx, length-1)].tolist()
                    action_list.append(a)

                save_dict['action'] = action_list
            
                data.append(save_dict)
                count += 1

print('done, data number:', count)
with open(f"data/libero/{task_suite_name}-states.json", 'w') as f:
    json.dump(data, f, indent=4)