
import json 
import os 
import numpy as np 

#p(Y | Z)
f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/ood_spurious_dict.json"


performance = {
    ("vqa_v2_train", "vqa_v2_id_val") : 90.9,
    ("vqa_vs_train", "vqa_vs_id_test") : 0,

    ("vqa_v2_train", "vqa_vs_KO") : 26.65, 
    ("vqa_v2_train", "vqa_vs_KOP") : 31.25,
    ("vqa_v2_train", "vqa_vs_KW") : 35.9, 
    ("vqa_v2_train", "vqa_vs_KW_KO") : 42.66, 
    ("vqa_v2_train", "vqa_vs_KWP") : 42.96, 
    ("vqa_v2_train", "vqa_vs_QT") : 27.45, 
    ("vqa_v2_train", "vqa_vs_QT_KO") : 37.78, 
    ("vqa_v2_train", "vqa_vs_QT_KW") : 44.68, 
    ("vqa_v2_train", "vqa_vs_QT_KW_KO") : 45.48, 

    ("vqa_v2_train", "vqa_vs_id_test") : 54.41, 
    ("vqa_v2_train", "vqa_ce_ood_val") : 79.81, 
    ("vqa_v2_train", "vqa_cp_ood_test") : 90.42, 

    ("vqa_v2_train", "textvqa_ood_val") : 44.25, 
    ("vqa_v2_train", "advqa_ood_val"): 56.86, 
    ("vqa_v2_train", "vqa_rephrasings_ood_val"): 85.55, 

    ("vqa_cp_train","vqa_cp_ood_test") : 0,

    ("vqa_vs_train", "vqa_vs_id_val") : 0,
    ("vqa_vs_train", "vqa_vs_id_test") : 0,
    
    ("vqa_vs_train", "vqa_vs_KO") : 0 ,
    ("vqa_vs_train", "vqa_vs_KOP") : 0 ,
    ("vqa_vs_train", "vqa_vs_KW") : 0 ,
    ("vqa_vs_train", "vqa_vs_KW_KO") : 0 ,
    ("vqa_vs_train", "vqa_vs_KWP") : 0 ,
    ("vqa_vs_train", "vqa_vs_QT") : 0 ,
    ("vqa_vs_train", "vqa_vs_QT_KO") : 0 ,
    ("vqa_vs_train", "vqa_vs_QT_KW") : 0 ,
    ("vqa_vs_train", "vqa_vs_QT_KW_KO") : 0 ,
    ("textvqa_train", "textvqa_ood_val") : 0
}

splits = [
    #train, test split 
    
    ("vqa_v2_train", "vqa_v2_id_val"),
    ("vqa_vs_train", "vqa_vs_id_test"), 


    #vqa_v2 train vs. vqa_vs 9 concepts 
    ("vqa_v2_train", "vqa_vs_KO") ,
    ("vqa_v2_train", "vqa_vs_KOP") ,
    ("vqa_v2_train", "vqa_vs_KW") ,
    ("vqa_v2_train", "vqa_vs_KW_KO") ,
    ("vqa_v2_train", "vqa_vs_KWP") ,
    ("vqa_v2_train", "vqa_vs_QT") ,
    ("vqa_v2_train", "vqa_vs_QT_KO") ,
    ("vqa_v2_train", "vqa_vs_QT_KW") ,
    ("vqa_v2_train", "vqa_vs_QT_KW_KO") , 

    ("vqa_v2_train", "vqa_vs_id_test"), 
    ("vqa_v2_train", "vqa_ce_ood_val"), 
    ("vqa_v2_train", "vqa_cp_ood_test"), 

    ("vqa_v2_train", "textvqa_ood_val"), 
    ("vqa_v2_train", "advqa_ood_val"), 
    ("vqa_v2_train", "vqa_rephrasings_ood_val"), 
    
    ("vqa_cp_train","vqa_cp_ood_test"), 
    ("vqa_vs_train", "vqa_vs_id_val"), 
    ("vqa_vs_train", "vqa_vs_id_test"), 
    
    ("vqa_vs_train", "vqa_vs_KO") ,
    ("vqa_vs_train", "vqa_vs_KOP") ,
    ("vqa_vs_train", "vqa_vs_KW") ,
    ("vqa_vs_train", "vqa_vs_KW_KO") ,
    ("vqa_vs_train", "vqa_vs_KWP") ,
    ("vqa_vs_train", "vqa_vs_QT") ,
    ("vqa_vs_train", "vqa_vs_QT_KO") ,
    ("vqa_vs_train", "vqa_vs_QT_KW") ,
    ("vqa_vs_train", "vqa_vs_QT_KW_KO") , 
    ("textvqa_train", "textvqa_ood_val")
]






#for each split pair -> measure across all key concepts and measure significance 

with open(f, 'r') as file : 
    data = json.load(file)

concept_list = ["KO","KOP","KW","KW+KO","KWP","QT","QT+KO","QT+KW","QT+KW+KO"]
plot_type = ["min", "max", "avg", "KO","KOP","KW","KW+KO","KWP","QT","QT+KO","QT+KW","QT+KW+KO"]

combined_ood_perf = { 
}

# print(data["vqa_v2_train"].keys())
print(data["vqa_v2_train"]["vqa_vs_KO"])
def get_comparable_value(value):
    return value[0] if isinstance(value, list) else value

for (train_split, test_split) in splits :
    print(f"Current split : ({train_split}, {test_split}) \n")
    val_dict = data[train_split][test_split]
    list_val = [] 

    for key, value in val_dict.items():
        if isinstance(value, list) or isinstance(value, tuple):    
            list_val.append(value[0])
        elif isinstance(value, (float, int)):
            list_val.append(value)
    print(list_val)
    #verify value type 
    max_key = max(val_dict, key=lambda k: get_comparable_value(val_dict[k]))
    max_value = val_dict[max_key][0] if isinstance(val_dict[max_key],list) else val_dict[max_key]
    list_val_no_zeros = [val for val in list_val if val != 0]
    print("", list_val)
    print("list no zeroes", list_val_no_zeros)

    avg_shift = np.mean(list_val_no_zeros)

    min_key = min(val_dict, key=lambda k: get_comparable_value(val_dict[k]))
    min_value = val_dict[min_key][0] if isinstance(val_dict[min_key],list) else val_dict[min_key]

    print(f"concept : {max_key},  max shift value : {max_value}")
    print(f"concept : {min_key},  min shift value : {min_value}")

    combined_ood_perf[(train_split, test_split)] = {} 
    combined_ood_perf[(train_split, test_split)]["perf"] = performance[(train_split, test_split)]
    combined_ood_perf[(train_split, test_split)]["max"] = (max_value, max_key)
    combined_ood_perf[(train_split, test_split)]["min"] = (min_value, min_key)
    combined_ood_perf[(train_split, test_split)]["avg"] = avg_shift

    for concept in concept_list : 
        if concept not in val_dict:
            combined_ood_perf[(train_split, test_split)][concept] = 0
        else : 
            if isinstance(val_dict[concept],list) : 
                val = val_dict[concept][0]
            else : 
                val = val_dict[concept]
            combined_ood_perf[(train_split, test_split)][concept] = val
            


    print(f"average shift across concepts: {avg_shift}\n") 

combined_ood_perf = {f"{k[0]},{k[1]}": v for k, v in combined_ood_perf.items()}

with open("/nethome/bmaneech3/flash/vlm_robustness/result_output/comb_shift_perf_dict.json", 'w') as file : 
    json.dump(combined_ood_perf , file)
