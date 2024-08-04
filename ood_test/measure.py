import json 
import numpy as np 
from collections import defaultdict
from heapq import nlargest
import os
"""

functions : 

1) count f(question_type, answer) 

2) calculate distance metric - between train & test split -> find way to optimize 
    - tensor 
"""
keyobj_file = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/coco/trainval_img_samples.json"
textvqa_obj_file = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/images/open_images_samples.json"


special_pair_ko = False 
with open(keyobj_file, 'r') as file : 
    keyobj_dict = json.load(file)


coco_id_2_label_file = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/coco/trainval_keyobj_samples.json"
with open(coco_id_2_label_file) as file : 
    coco_id_2_label = json.load(file)

with open(textvqa_obj_file, 'r') as file : 
    textvqa_obj_dict = json.load(file)


def count_keywords_obj(file_name) : 
    #return the f(w), f(a), f(w,a) dictionary 

    f_w = defaultdict(int)
    f_a = defaultdict(int)
    f_w_a = defaultdict(int)
    f_obj = defaultdict(int)
    f_obj_a = defaultdict(int)
    

    with open(file_name, 'r') as file : 
        data = json.load(file)

        # k = len(data) #number of samples 
        for sample in data : 


            if sample["question_type"] != None : 
                question_type = sample["question_type"].lower()
                question = sample["question"].lower()
                #remove question type part of the question 
                idx = question.find(question_type)
                if idx != -1:
                    question = question[idx+len(question_type):]

            else : #could deduct k% of the start of question??
                question = sample["question"].lower()

            ans= sample["answer"].lower()
            img_id = sample["image_id"]


            #avoid duplicates for now 
            words = set(question.split())

            for word in words:
                f_w[word] += 1
                f_w_a[(word, ans)] += 1

            f_a[ans] += 1

            if "textvqa" in file_name : 
                categories = set(textvqa_obj_dict[img_id]["image_classes"])

            else : 

                #count objects 
                if str(img_id) not in keyobj_dict : 
                    categories = set() 
                    
                elif special_pair_ko : 
                    categories = set([coco_id_2_label[str(label_id)]["name"].lower() for label_id in keyobj_dict[str(img_id)]["categories"]]) #becareful of types when read from json file 

                else :
                    categories = set(keyobj_dict[str(img_id)]["categories"])
            for cat in categories : 
                f_obj[cat] += 1 
                f_obj_a[(cat, ans)] += 1


    # print("verify length")
    # print("length fw", len(f_w))


    return f_w, f_a, f_w_a, f_obj, f_obj_a


def select_keyword_obj(file_name) : 
    #ref global f_w, f_a, ... 

    print("length of f_w", len(f_w))
    # print("keyobjdict ", keyobj_dict)
    
    kw = [] 
    kwp = [] 
    ko = [] 
    kop = [] 
    qt = [] 

    co = 0 
    with open(file_name, 'r') as file : 
        data = json.load(file)

        k = len(data) #number of samples 
        for sample in data : 
    
            kw_sample = {}
            kwp_sample = {}
            ko_sample = {}
            kop_sample = {}
            qt_sample = {} 
            
            
            if sample["question_type"] != None : 
                question_type = sample["question_type"].lower()
                question = sample["question"].lower()

                #remove question type part of the question 
                idx = question.find(question_type)
                if idx != -1:
                    question = question[idx+len(question_type):]
                
            else :
                question_type = None
                question = sample["question"].lower()
            
        
            ans = sample["answer"].lower()
            img_id = sample["image_id"]
            qid = sample["question_id"]
            qt_sample['qid'] = qid
            kw_sample['qid'] = qid
            kwp_sample['qid'] = qid
            ko_sample['qid'] = qid
            kop_sample['qid'] = qid
            qt_sample['ans'] = ans
            kw_sample['ans'] = ans
            kwp_sample['ans'] = ans
            ko_sample['ans'] = ans
            kop_sample['ans'] = ans

            qt_sample["concept"] = question_type

            #avoid duplicates for now 
            words = set(question.split())

            if len(words) > 0 : 
            
                mi_scores = defaultdict(float)
                #select 2 word with highest MI (w, ans)
                for word in words : 
                    # co = np.log(f_w_a[(word, ans)]/((f_w[word] * f_a[ans])/k))
                    try : 
                        mi = np.log(f_w_a[(word, ans)] / ((f_w[word] * f_a[ans]) / k))

                    except Exception as e : 
                        raise e 

                    mi_scores[word]= mi
                
                top_words = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:2]

                if len(top_words) == 2 : 
                    word_1, word_2 = top_words[0][0], top_words[1][0]

                else : 
                    word_1, word_2 = top_words[0][0], None

            else : 
                word_1, word_2 = None, None 

            kw_sample["concept"] = word_1
            kwp_sample["concept"] = tuple([word_1, word_2])
            
            #find key object 

            #handle this case 
            if "textvqa" in file_name : 
                objs = set(textvqa_obj_dict[img_id]["image_classes"])

            else : 

                #count objects 
                if str(img_id) not in keyobj_dict : 
                    objs = set() 
                    
                elif special_pair_ko : 
                    objs = set([coco_id_2_label[str(label_id)]["name"].lower() for label_id in keyobj_dict[str(img_id)]["categories"]]) #becareful of types when read from json file 

                else :
                    objs = set(keyobj_dict[str(img_id)]["categories"])


            if len(objs) > 0 : 
                # print(objs)
                mi_scores = defaultdict(float)

                for obj in objs :
                    try  :
                        mi = np.log(f_obj_a[(obj, ans)] / ((f_obj[obj] * f_a[ans]) / k))
                    except Exception as e : 
                        print("obj: ", obj, "ans: ", ans, "fobj ", f_obj[obj], "f_obj_a ", f_obj_a[(obj,ans)])
                        raise e 
                    mi_scores[obj]= mi

                top_objs = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                # print("top_objs", top_objs)
                
                if len(top_objs) == 2 : 
                    obj_1, obj_2 = top_objs[0][0], top_objs[1][0]
                else : 
                    # print(top_objs)
                    # print(img_id)
                    obj_1 , obj_2 = top_objs[0][0], None 


            else : 
                # print(img_id)
                obj_1 = None 
                obj_2 = None 
                co += 1 
            ko_sample["concept"] = obj_1
            kop_sample["concept"] = tuple([obj_1, obj_2])


            kw.append(kw_sample)
            kwp.append(kwp_sample)
            ko.append(ko_sample)
            kop.append(kop_sample)
            qt.append(qt_sample)

    print("total samples ",k )
    print("no keyobj count ", co)
    return kw, kwp, ko, kop, qt

    #return list of dictionary 
    #return samples each : [
    # {
    # question_id, keyword, answer
    # } ] 

    #then merge them later 

def count_concept_occurence(file_name, type ="QT") : 
    #make sure f_w, ... global
    
    #for a filename - count concept occurence for specific list of types 
    global f_w, f_a, f_w_a, f_obj, f_obj_a 
    global kw, kwp, ko, kop, qt
    f_w, f_a, f_w_a, f_obj, f_obj_a = count_keywords_obj(file_name)
    kw, kwp, ko, kop, qt = select_keyword_obj(file_name)

    select_concept = {
        "KW" : kw, 
        "KWP" : kwp, 
        "KO" : ko, 
        "KOP" : kop, 
        "QT" : qt
    } 
    

    concept_list = [] 

    if type == "KO" : 
        concept_list.append("KO")

    elif type == "KOP" : 
        concept_list.append("KOP")
    
    elif type == "KW" : 
        concept_list.append("KW") 

    elif type == "KW+KO" : 
        concept_list.append("KW")
        concept_list.append("KO")

    elif type == "KWP" : 
        concept_list.append("KWP")
    elif type == "QT" : 
        concept_list.append("QT")

    elif type == "QT+KO" : 
        concept_list.append("QT")
        concept_list.append("KO")

    elif type == "QT+KW" : 
        concept_list.append("QT")
        concept_list.append("KW")

    elif type == "QT+KW+KO" : 
        concept_list.append("QT")
        concept_list.append("KW")
        concept_list.append("KO")


    else : 
        raise Exception("Wrong concept type param")
        
    merged_dict = {} #ref by qid 
    total_samples = 0 

    qid_samples = set()


    concept_list = tuple(concept_list)


    for concept in concept_list : 
        # print(concept)

        samples = select_concept[concept]
        # print(samples)
        # break
        total_samples = len(samples)
        
        for sample in samples : 
            """
            sample : 
            { 
            'qid' : int , 
            'concept' : type, 
            'answer' : str 
            }
            """
            qid = sample['qid']
            if qid not in merged_dict : 
                merged_dict[qid] = {
                    'qid': qid,
                    'concepts': tuple(),
                    'answer': sample['ans']
                }
                qid_samples.add(qid)

            merged_dict[qid]["concepts"] += (sample["concept"],)
    


    # merged_list = list(merged_dict.values())

    #compute freq table 

    freq_table = defaultdict(int) #key : (concept, ans)
    concept_total = defaultdict(int)



    for qid, sample in merged_dict.items() : 
        concepts = sample["concepts"] #list of concept values (which can be of any type)
        # print(concepts)
        ans = sample["answer"]
        if len(ans) <= 0 : 
            raise Exception("Found invalid answer string")
        freq_table[(concepts, ans)] += 1 
        concept_total[concepts] += 1 

    #create prob table 
    prob_table = defaultdict(dict)
    for (concepts, ans), freq in freq_table.items() : 
        prob_table[concepts][ans] = freq / concept_total[concepts]

    return prob_table, total_samples, qid_samples 


# def compute 

    

def hellinger_dist(train_dist, test_dist) : 
    concept_keys = set(train_dist.keys()).union(set(test_dist.keys())) #keys = concepts 

    #train_dist (concept, ans)

    #find average of shift metric across all possible joint conditional prob dist 

    metric_dict = {}

    for key in concept_keys:
        train_value = train_dist.get(key, {}) 
        test_value = test_dist.get(key, {}) 

        """
        train_value = {
        ans1 : prob1 ,
        ans2 : prob2, 
        ans3 : prob3 , ... 
        }
        """
        metric = 0

        #add small epsilon? 
        eps = 1e-8

        all_ans_keys = set(train_value.keys()).union(set(test_value.keys()))

        for ans in all_ans_keys : 
            train_prob = train_value.get(ans, 0) + eps 
            test_prob = test_value.get(ans, 0) + eps

            metric += (np.sqrt(train_prob) - np.sqrt(test_prob)) ** 2 

        #for this concept 
        metric = 1/np.sqrt(2) * np.sqrt(metric)
        metric_dict[key] = metric  


    #find average metric, sd 

    avg_metric = np.mean(list(metric_dict.values()))

    sd_metric = np.std(list(metric_dict.values()))
    # print(metric_dict)

    return avg_metric, sd_metric


def jaccard_similarity(set_a, set_b) : 
    #question_id 
    
    # set_a = set(split_a)
    # set_b = set(split_b)

    intersect = set_a.intersection(set_b)

    union = set_a.union(set_b)

    if len(union) == 0:
        return 0
    return len(intersect) / len(union)



vqa_vs_train_file = "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/train/train_combined_ann.json"
# splits = {
#     # "vqa_v2_train , vqa_v2_val" : ("/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_train2014_annotations.json", "/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_val2014_annotations.json"),
#     # "vqa_v2_train, vqa_cp_test" : ("/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_train2014_annotations.json", "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json"), 
#     # "vqa_cp_train, vqa_cp_test" : ("/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_annotations.json", "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json"), 

#     # "vqa_vs_train, vqa_vs_test_QT" : ("/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_annotations.json","/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ans.json"),
#     # "vqa_v2_train, vqa_vs_test_qt" : ("/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_train2014_annotations.json", "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ans.json") 

   
# }
# splits = {
#     "KO": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/KO_combined_ann.json"),
#     "KOP": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/KOP_combined_ann.json"),
#     "KW": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/KW_combined_ann.json"),
#     "KW+KO": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/KW_KO_combined_ann.json"),
#     "KWP": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/KWP_combined_ann.json"),
#     "QT": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/QT_combined_ann.json"),
#     "QT+KO": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/QT_KO_combined_ann.json"),
#     "QT+KW": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/QT_KW_combined_ann.json"),
#     "QT+KW+KO": (vqa_vs_train_file, "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/QT_KW_KO_combined_ann.json")
# }




splits = [
    #vqav2 train & vqav2 val 
    ("KO", ["vqa_v2", "train"], ["vqa_v2", "id_val"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_v2", "id_val"]),
    ("KW", ["vqa_v2", "train"], ["vqa_v2", "id_val"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_v2", "id_val"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_v2", "id_val"]),
    ("QT", ["vqa_v2", "train"], ["vqa_v2", "id_val"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_v2", "id_val"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_v2", "id_val"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_v2", "id_val"]),
    
    # #vqav2 train & vs. vqavs id test 
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "id_test"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "id_test"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "id_test"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "id_test"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "id_test"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "id_test"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "id_test"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "id_test"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "id_test"]),

    # #vqa v2 train vs. vqavs 9 concepts (OOD test)

    # #vqa v2 train vs. vqavs KO 
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "KO"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "KO"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "KO"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KO"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "KO"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "KO"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "KO"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "KO"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KO"]),

    # #vqa v2 train vs. vqavs KOP
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "KOP"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "KOP"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "KOP"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KOP"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "KOP"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "KOP"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "KOP"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "KOP"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KOP"]),


    # #vqa v2 train vs. vqavs KW
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "KW"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "KW"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "KW"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KW"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "KW"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "KW"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "KW"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "KW"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KW"]),

    # #vqa v2 train vs. vqavs KW_KO
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KW_KO"]),

    # #vqa v2 train vs. vqavs KWP
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "KWP"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "KWP"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "KWP"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KWP"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "KWP"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "KWP"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "KWP"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "KWP"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "KWP"]),

    # #vqa v2 train vs. vqavs QT
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "QT"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "QT"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "QT"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "QT"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "QT"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "QT"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "QT"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "QT"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "QT"]),


    # #vqa v2 train vs. vqavs QT_KO
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KO"]),

    # #vqa v2 train vs. vqavs QT_KW
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KW"]),


    # #vqa v2 train vs. vqavs QT_KW_KO
    ("KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("KW", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("QT", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_vs", "QT_KW_KO"]),


    # #vqa v2 train vs. vqa ce ood val 
    ("KO", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]),
    ("KW", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]),
    ("QT", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_ce", "ood_val"]),

    # #vqa v2 train vs. vqa cp ood test
    ("KO", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]),
    ("KW", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]),
    ("QT", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_cp", "ood_test"]),


    #vqa v2 train vs. textvqa ood_val 
    ("KO", ["vqa_v2", "train"], ["textvqa", "ood_val"]), 
    ("KOP", ["vqa_v2", "train"], ["textvqa", "ood_val"]),
    ("KW", ["vqa_v2", "train"], ["textvqa", "ood_val"]),
    ("KW+KO", ["vqa_v2", "train"], ["textvqa", "ood_val"]),
    ("KWP", ["vqa_v2", "train"], ["textvqa", "ood_val"]),
    # ("QT", ["vqa_v2", "train"], ["textvqa", "ood_val"]),
    # ("QT+KO", ["vqa_v2", "train"], ["textvqa", "ood_val"]),
    # ("QT+KW", ["vqa_v2", "train"], ["textvqa", "ood_val"]),
    # ("QT+KW+KO", ["vqa_v2", "train"], ["textvqa", "ood_val"]),


    #vqav2 train vs. advqa ood val
    ("KO", ["vqa_v2", "train"], ["advqa", "ood_val"]), 
    ("KOP", ["vqa_v2", "train"], ["advqa", "ood_val"]),
    ("KW", ["vqa_v2", "train"], ["advqa", "ood_val"]),
    ("KW+KO", ["vqa_v2", "train"], ["advqa", "ood_val"]),
    ("KWP", ["vqa_v2", "train"], ["advqa", "ood_val"]),
    # ("QT", ["vqa_v2", "train"], ["advqa", "ood_val"]),
    # ("QT+KO", ["vqa_v2", "train"], ["advqa", "ood_val"]),
    # ("QT+KW", ["vqa_v2", "train"], ["advqa", "ood_val"]),
    # ("QT+KW+KO", ["vqa_v2", "train"], ["advqa", "ood_val"]),

    #vqav2 train vs. vqa_rephrasings 
    ("KO", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]), 
    ("KOP", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]),
    ("KW", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]),
    ("KW+KO", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]),
    ("KWP", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]),
    ("QT", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]),
    ("QT+KO", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]),
    ("QT+KW", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]),
    ("QT+KW+KO", ["vqa_v2", "train"], ["vqa_rephrasings", "ood_val"]),


    ## vqa cp train vs. vqa cp ood test 
    ("KO", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]), 
    ("KOP", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]),
    ("KW", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]),
    ("KW+KO", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]),
    ("KWP", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]),
    ("QT", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]),
    ("QT+KO", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]),
    ("QT+KW", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]),
    ("QT+KW+KO", ["vqa_cp", "train"], ["vqa_cp", "ood_test"]),


    # #vqavs train vs. vqavs id test 
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "id_test"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "id_test"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "id_test"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "id_test"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "id_test"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "id_test"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "id_test"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "id_test"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "id_test"]),

    # #vqavs train vs. vqavs id val 
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "id_val"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "id_val"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "id_val"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "id_val"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "id_val"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "id_val"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "id_val"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "id_val"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "id_val"]),

    #vqavs train vs. 9 concepts (OOD Test) """================================"""
    #===========================================================================

    # #vqa v2 train vs. vqavs KO 
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "KO"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "KO"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "KO"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KO"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "KO"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "KO"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "KO"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "KO"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KO"]),

    # #vqa v2 train vs. vqavs KOP
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "KOP"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "KOP"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "KOP"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KOP"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "KOP"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "KOP"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "KOP"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "KOP"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KOP"]),


    # #vqa v2 train vs. vqavs KW
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "KW"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "KW"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "KW"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KW"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "KW"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "KW"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "KW"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "KW"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KW"]),

    # #vqa v2 train vs. vqavs KW_KO
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KW_KO"]),

    # #vqa v2 train vs. vqavs KWP
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "KWP"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "KWP"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "KWP"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KWP"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "KWP"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "KWP"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "KWP"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "KWP"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "KWP"]),

    # #vqa v2 train vs. vqavs QT
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "QT"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "QT"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "QT"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "QT"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "QT"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "QT"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "QT"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "QT"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "QT"]),


    # #vqa v2 train vs. vqavs QT_KO
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KO"]),

    # #vqa v2 train vs. vqavs QT_KW
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KW"]),


    # #vqa v2 train vs. vqavs QT_KW_KO
    ("KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]), 
    ("KOP", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("KW", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("KW+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("KWP", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("QT", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("QT+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("QT+KW", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]),
    ("QT+KW+KO", ["vqa_vs", "train"], ["vqa_vs", "QT_KW_KO"]),

    #textvqa train vs. textvqa test 
    ("KO", ["textvqa", "train"], ["textvqa", "ood_val"]), 
    ("KOP", ["textvqa", "train"], ["textvqa", "ood_val"]),
    ("KW", ["textvqa", "train"], ["textvqa", "ood_val"]),
    ("KW+KO", ["textvqa", "train"], ["textvqa", "ood_val"]),
    ("KWP", ["textvqa", "train"], ["textvqa", "ood_val"]),
    # ("QT", ["textvqa", "train"], ["textvqa", "ood_val"]),
    # ("QT+KO", ["textvqa", "train"], ["textvqa", "ood_val"]),
    # ("QT+KW", ["textvqa", "train"], ["textvqa", "ood_val"]),
    # ("QT+KW+KO", ["textvqa", "train"], ["textvqa", "ood_val"])
]


results_file = "/nethome/bmaneech3/flash/vlm_robustness/result_output/ood_spurious_dict.json"
jacc_file = "/nethome/bmaneech3/flash/vlm_robustness/result_output/jaccard_dict.json"
with open(results_file, 'r') as file : 
    results_dict = json.load(file) #read from results dict 

with open(jacc_file, 'r') as file : 
    jaccard_dict = json.load(file)

root_dir = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/full_ds"
id = 1 
for split in splits : 

    concept_type = split[0]
    train_ds, train_split = split[1][0], split[1][1]
    test_ds, test_split = split[2][0], split[2][1]


    title = f"{concept_type} : {train_ds}_{train_split}, {test_ds}_{test_split}" 
    print("title", title)

    if (f"{train_ds}_{train_split}" in results_dict): 
        if (f"{test_ds}_{test_split}" in results_dict[f"{train_ds}_{train_split}"]) : 
            if (concept_type in results_dict[f"{train_ds}_{train_split}"][f"{test_ds}_{test_split}"]) : 
                print(f"{title} already measured")
                continue  

    id += 1 
    print(f"\n Measure {id}: {title}")
    if not(train_ds == "textvqa" and test_ds == "textvqa") : 
        special_pair_ko = True 
    else : 
        special_pair_ko = False


    train_split_file= os.path.join(root_dir, train_ds, f"{train_split}_combined_ann.json")
    test_split_file  = os.path.join(root_dir, test_ds, f"{test_split}_combined_ann.json")
   

    train_dist, n_train, samples_train = count_concept_occurence(train_split_file, type=concept_type)
    test_dist, n_test, samples_test = count_concept_occurence(test_split_file, type=concept_type)

    shift_res = hellinger_dist(train_dist, test_dist)
    jacc_sim = jaccard_similarity(samples_train, samples_test)
    print("Result Hellinger dist: ", shift_res)
    print("Jaccard Similarity: ", jacc_sim)


    train_split = f"{train_ds}_{train_split}"
    test_split = f"{test_ds}_{test_split}"

    if train_split in results_dict : 
        if test_split not in results_dict[train_split] : 
            results_dict[train_split][test_split] = {} 
        results_dict[train_split][test_split][concept_type] = shift_res
        
    else : 
        results_dict[train_split] = {}
        results_dict[train_split][test_split] = {} 
        results_dict[train_split][test_split][concept_type] = shift_res 

    if train_split not in jaccard_dict: 
        jaccard_dict[train_split] = {} 

    jaccard_dict[train_split][test_split] = jacc_sim


with open(results_file, 'w') as file : 
    json.dump(results_dict, file)

with open(jacc_file, 'w') as file : 
    json.dump(results_dict, file)





