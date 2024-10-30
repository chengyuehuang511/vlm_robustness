import torch 
from collections import defaultdict
f = "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/v2_OpenEnded_mscoco_val2014_questions.json"
import json
combine_path = "/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/ds_split_2_file.json"
with open(combine_path, 'r') as file : 
    ds_split_2_file = json.load(file)


    for split, f in ds_split_2_file.items() : 
        if "vqa_vs" in split or "vqa_lol" in split : 
            continue 
        
        print("Split", split )

        with open(f, 'r') as file : 
            data = json.load(file)
            print("len", len(data))
            ques_dict  = defaultdict(list)
            print(data[0].keys())
            print(data[0]["question_id"])
            if "image_id" in data[0] : 
                print(data[0]["image_id"])
            if "image" in data[0] : 
                print(data[0]["image"])

        print()
#     co = 0 
#     for sample in data : 
#         if sample["question_id"] in ques_dict and sample["image_id"] not in ques_dict[sample["question_id"]]: 
#             # print("yess") 
#             co += 1 
        
#         ques_dict[sample["question_id"]].append(sample["image_id"])
    

#     print(len(ques_dict))

#     print(co)

# # print(d.size())

f =  "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json"
with open(f, 'r') as file : 
    data = json.load(file)
    ques_dict  = defaultdict(list)
    print(data[0].keys())
    print(data[0]["question_id"])
    if "image_id" in data[0] : 
        print(data[0]["image_id"])
    if "image" in data[0] : 
        print(data[0]["image"])




# f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/lora/coco_advqa_val_image_joint.pth"
# data = torch.load(f)

# print(data.keys())