import pandas as pd 
from tqdm import tqdm 
import json 
    
label_map = pd.read_csv("/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/2017_11/class-descriptions.csv")
label_map = label_map.to_dict(orient='records')


# print(len(label_map))

file_list = ["/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/2017_11/test/annotations-human.csv",
             "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/2017_11/train/annotations-human.csv", 
             "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/2017_11/validation/annotations-human.csv"
             ]

#create image samples 
df = pd.read_csv(file_list[0])
print(df.columns)


# for file in file_list : 
#     id_2_images = {} #image id -> class list 



#     df = pd.read_csv()






# f = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/val/combined_data.json"

# with open(f, 'r') as combined_data : 
#     combined_data = json.load(combined_data) 


#     for samples in combined_data 
