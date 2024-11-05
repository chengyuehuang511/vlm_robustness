import torch 
import json 
import os 

import numpy as np 
import matplotlib.pyplot as plt

# for each measurement -> get the OOD and ID threshold region 
#want instance id from the evaluation 


"""
- plot histogram distribution 
- get_range_samples(left_range, right_range, concept) -> List[instance ids]
- process : left, peak, right region range 

- get incorrect samples plot 

- intersect -> (ID image samples) & (OOD question samples)
- count left & right tail (ID,ID)

"""
train_split = "coco_vqav2_train_val"

def plot_histogram(title, output_file_path, test_split, train_score_dict, test_score_dict, concept) : 

    """
    Args : 

    title (string)
    output_file_path (string)
    train_dict (dict) : 
        split {
            instance_id : {
                image : 0 
                joint : 0 
                question : 0  #modify to add question within the instance 
            }
        }
    """

    train_scores = np.array([value[concept] for key, value in train_score_dict[]])
    test_scores = np.array([value[concept] for key, value in test_score_dict[test_split]])

    n1, bin_edges, _ = plt.hist(datasets[0][c].numpy(), density=True, bins=100, alpha=0.5, label=f'{train_split}')
    n2, bin_edges, _ = plt.hist(datasets[1][c].numpy(), density=True, bins=100, alpha=0.5, label=f'{test_split}')





    
























