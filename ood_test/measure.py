import json 
import numpy as np 
from collections import defaultdict
from heapq import nlargest

"""

functions : 

1) count f(question_type, answer) 

2) calculate distance metric - between train & test split -> find way to optimize 
    - tensor 
"""
keyobj_file = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/sample_org/trainval_img_samples.json"

keyobj_dict = None 



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

            question_type = sample["question_type"].lower()
            question = sample["question"].lower()
            ans= sample["answer"].lower()
            img_id = sample["image_id"]

            #remove question type part of the question 
            idx = question.find(question_type)
            if idx != -1:
                question = question[idx+len(question_type):]

            #avoid duplicates for now 
            words = set(question.split())

            for word in words:
                f_w[word] += 1
                f_w_a[(word, ans)] += 1

            f_a[ans] += 1

            #count objects 

            if str(img_id) not in keyobj_dict : 
                categories = set() 

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
            
            
            question_type = sample["question_type"].lower()
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
            
            #remove question type part of the question 
            idx = question.find(question_type)
            if idx != -1:
                question = question[idx+len(question_type):]

            #avoid duplicates for now 
            words = set(question.split())
            
            mi_scores = defaultdict(float)
            #select 2 word with highest MI (w, ans)
            for word in words : 
                # co = np.log(f_w_a[(word, ans)]/((f_w[word] * f_a[ans])/k))
                try : 
                    mi = np.log(f_w_a[(word, ans)] / ((f_w[word] * f_a[ans]) / k))

                except Exception as e : 
                    # print(question)
                    # print(question_type)
                    # print(words)
                    # print("word: ", word, "fw ", f_w, "f_w_a ", f_w_a)
                    
                    raise e 

                mi_scores[word]= mi
            
            top_words = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:2]

            if len(top_words) == 2 : 
                word_1, word_2 = top_words[0][0], top_words[1][0]

            else : 
                word_1, word_2 = top_words[0][0], None


            kw_sample["concept"] = word_1
            kwp_sample["concept"] = tuple([word_1, word_2])
            
            #find key object 


            if str(img_id) in keyobj_dict : 
                objs = set(keyobj_dict[str(img_id)]["categories"])
                # print(objs)
                mi_scores = defaultdict(float)

                for obj in objs :
                    try  :
                        mi = np.log(f_obj_a[(obj, ans)] / ((f_obj[obj] * f_a[ans]) / k))
                    except Exception as e : 
                        # print("obj: ", obj, "fobj ", f_obj, "f_obj_a ", f_obj_a)
                        raise e 
                    mi_scores[obj]= mi

                top_objs = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                
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

        all_ans_keys = set(train_value.keys()).union(set(test_value.keys()))

        for ans in all_ans_keys : 
            train_prob = train_value.get(ans, 0)
            test_prob = test_value.get(ans, 0)

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



with open(keyobj_file, 'r') as file : 
    keyobj_dict = json.load(file)


vqa_vs_train_file = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/train_combined_ann.json"
# splits = {
#     # "vqa_v2_train , vqa_v2_val" : ("/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_train2014_annotations.json", "/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_val2014_annotations.json"),
#     # "vqa_v2_train, vqa_cp_test" : ("/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_train2014_annotations.json", "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json"), 
#     # "vqa_cp_train, vqa_cp_test" : ("/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_annotations.json", "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json"), 

#     # "vqa_vs_train, vqa_vs_test_QT" : ("/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_annotations.json","/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ans.json"),
#     # "vqa_v2_train, vqa_vs_test_qt" : ("/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_train2014_annotations.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ans.json") 

    
   
# }
splits = {
    "KO": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/KO_combined_ann.json"),
    "KOP": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/KOP_combined_ann.json"),
    "KW": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/KW_combined_ann.json"),
    "KW+KO": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/KW_KO_combined_ann.json"),
    "KWP": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/KWP_combined_ann.json"),
    "QT": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/QT_combined_ann.json"),
    "QT+KO": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/QT_KO_combined_ann.json"),
    "QT+KW": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/QT_KW_combined_ann.json"),
    "QT+KW+KO": (vqa_vs_train_file, "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/QT_KW_KO_combined_ann.json")
}



for title in splits : 
    print(f"\n Measure : {title}")
    train_split, test_split = splits[title]
    # train_split = "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_annotations.json"

    # test_split = "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json"


    train_dist, n_train, samples_train = count_concept_occurence(train_split, type=title)
    test_dist, n_test, samples_test = count_concept_occurence(test_split, type=title)



    print(len(train_dist))

    print("Result Hellinger dist: ", hellinger_dist(train_dist, test_dist))
    print("Jaccard Similarity: ", jaccard_similarity(samples_train, samples_test))




# def count_occurence(file_name, type="QT") : 

#     freq_table = {}

#     with open(file_name, 'r') as file : 
#         data = json.load(file)

#         """{
#             "question_id" : int,
#             "image_id" : int,
#             "question_type" : str,
#             "answer_type" : str,
#             "answers" : [answer],
#             "multiple_choice_answer" : str
#             }
#         """


#         samples = set()  


#         if isinstance(data, dict): 
#             data = data["annotations"]


#         total_samples = len(data)

#         for sample in data: 
#             question_type = sample["question_type"]
#             answer = sample["multiple_choice_answer"]
#             id = sample["question_id"]


#             if id in samples : 
#                 raise Exception("Found duplicate ques id")
            
#             samples.add(id)

#             if len(answer) <= 0 : 
#                 raise Exception("Found invalid answer string")


#             if type == "QT"  :
#                 if (question_type, answer) not in freq_table : 
#                     freq_table[(question_type, answer)] = 1 

#                 else : 
#                     freq_table[(question_type, answer)] += 1 

#             elif type == "KW" or type == "KWP": 
#                 #find the keyword  


#     return freq_table, total_samples, samples