#modifies dataset and save with all necessary info needed per sample 
import json 
import os 
from tqdm import tqdm
from collections import Counter


def org_full_ds(file_list, ds_name) : 

    for (split_name, ques_file, ann_file) in file_list : 
        dup_question_id = {} 

        #create question_id -> question map 
        try : 
            co = 0 
            quesid_2_ques  = {} 
            #ref question id <-> annotations 
            with open(ques_file, 'r') as file :
                data = json.load(file)

                if isinstance(data, dict) : 
                    data = data["questions"] 

                for sample in data:
                    
                    if sample["question_id"] in quesid_2_ques : 
                      
                        dup_question_id[sample["question_id"]] = [sample["question"], quesid_2_ques[sample["question_id"]]]

                    quesid_2_ques[sample["question_id"]] = sample["question"]

            combined_list = []  


            print(f"total duplicates {co}")
            print(f"total samples {len(data)}")

            with open(ann_file, 'r') as file : 

                data = json.load(file)
                if isinstance(data, dict) : 
                    data = data["annotations"] 

                print("length of the data", len(data))

                for sample in data:
                    print("sampels")
                    #if multiple choice not found 

                    if 'multiple_choice_answer' not in sample : 
                        if 'answer' in sample : 
                            mc_ans = Counter(sample['answer']).most_common(1)[0][0]

                        elif 'answers' in sample : 
                            ans_list = [elem['answer'] for elem in sample['answers']]
                            mc_ans =  Counter(ans_list).most_common(1)[0][0]

                    else : 
                        mc_ans = sample['multiple_choice_answer']
                    
                    ques_id = sample["question_id"]
                    if ques_id in dup_question_id : 
                        print(f"ans {sample['multiple_choice_answer']} : {dup_question_id[ques_id][0]}, {dup_question_id[ques_id][1]}")

                        user_inp = input()

                        if int(user_inp) not in [0,1] :
                            print(f"image id : {sample['image_id']}")
                            user_inp = input()



                        question = dup_question_id[ques_id][int(user_inp)]

                    
                    else : 
                        question = quesid_2_ques[ques_id]

                    # answer_list = [d["answer"] for d in sample["answers"]]
                    #verify image_id is int 

                    # if not isinstance(sample["image_id"], int) :
                    #     # print(sample["img_id"])
                    #     # raise Exception("Image id not int")
                    #     continue 

                    if "question_type" not in sample : 
                        qt = None
                    else : 
                        qt = sample["question_type"]

                    if qt == "unknown" : 
                        qt = None
                    combined_sample = {
                        "question_id" : sample["question_id"] , 
                        "question" :question , 
                        "question_type" : qt, 
                        "answer" : mc_ans,
                        "image_id" : sample["image_id"], #must be int 
                        "dataset" : "vqa"
                    }

                    combined_list.append(combined_sample)
        
        except Exception as e : 
            print(f"split {split_name}, dataset_name {ds_name}")
            raise e
        
        print("comb list len", len(combined_list))
        dir_path = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/full_ds"
        comb_path = os.path.join(dir_path, ds_name, f"{split_name}_combined_ann.json")
        print(comb_path)
        with open(comb_path, 'w') as file : 
            json.dump(combined_list, file)


if __name__ == "__main__" : 

    dataset_split_map = {
        # "vqa_cp" : [("train", "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqacp2/raw/annotations/vqacp_v2_train_questions.json", "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqacp2/raw/annotations/vqacp_v2_train_annotations.json"), 
        # ("ood_test", "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqacp2/raw/annotations/vqacp_v2_test_questions.json", "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqacp2/raw/annotations/vqacp_v2_test_annotations.json")], 

        # "vqa_v2" : [ 
        #     ("train", "/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json", "/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_train2014_annotations.json"), 
        #     ("id_val", "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json", "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_mscoco_val2014_annotations.json") 
        #     # ("ID_test","/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2", "/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2")
        # ], 
        # "vqa_ce" : [
        #     ("ood_val", "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/question_subset.json", "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/annotation_subset.json")
        # ], 
        # "vqa_vs" : [("train", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_questions.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_annotations.json"), 
        # ("id_val", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_questions.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_annotations.json"), 
        # ("id_test", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_questions.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_annotations.json"), 
        # ("KO", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/OOD-Test-KO-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/OOD-Test-KO-Ans.json"),
        # ("KOP", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/OOD-Test-KOP-Ques.json","/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/OOD-Test-KOP-Ans.json"), 
        # ("KW", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/OOD-Test-KW-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/OOD-Test-KW-Ans.json"), 
        # ("KW_KO", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/OOD-Test-KW+KO-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/OOD-Test-KW+KO-Ans.json"), 
        # ("KWP", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/OOD-Test-KWP-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/OOD-Test-KWP-Ans.json"), 
        # ("QT", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ans.json"), 
        # ("QT_KO", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/OOD-Test-QT+KO-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/OOD-Test-QT+KO-Ans.json"), 
        # ("QT_KW", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/OOD-Test-QT+KW-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/OOD-Test-QT+KW-Ans.json"), 
        # ("QT_KW_KO", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/OOD-Test-QT+KW+KO-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/OOD-Test-QT+KW+KO-Ans.json")],
        # "vqa_rephrasings" : [
        #     ("ood_val", "/srv/datasets/vqa_rephrasings/v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json", "/srv/datasets/vqa_rephrasings/v2_mscoco_valrep2014_humans_og_annotations.json")
        # ]

#         "advqa" : [
#             ("ood_val", "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/v1_OpenEnded_mscoco_val2017_advqa_questions.json",
#    "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/v1_mscoco_val2017_advqa_annotations_new.json")
#         ]

        "textvqa" : [ 
            ("ood_val", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/val/question.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/val/annotation.json"),
            ("train","/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/train/question.json","/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/train/annotation.json")
        ]

    }



    for dataset_name,file_list in dataset_split_map.items() : 
        org_full_ds(file_list, dataset_name) 









