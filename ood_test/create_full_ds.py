#modifies dataset and save with all necessary info needed per sample 
import json 
import os 
from tqdm import tqdm


def org_full_ds(file_list) : 

    for (split_name, ques_file, ann_file) in file_list : 


        #create question_id -> question map 

        quesid_2_ques  = {} 
        #ref question id <-> annotations 
        with open(ques_file, 'r') as file :

            if isinstance(data, dict) : 
                data = data["questions"] 

            
            data = json.load(file)
            for sample in tqdm(data, desc=f'Loading {split_name} questions', unit=' samples'):
                
                if sample["question_id"] in quesid_2_ques : 
                    raise Exception("duplicate question id error")
                
                quesid_2_ques[sample["question_id"]] = sample["question"]


        combined_list = []  

        with open(ann_file, 'r') as file : 

            data = json.load(file)
            if isinstance(data, dict) : 
                data = data["annotations"] 

            for sample in tqdm(data, desc=f'Processing {split_name}', unit=' samples'):
                
                ques_id = sample["question_id"]
                question = quesid_2_ques[ques_id]

                # answer_list = [d["answer"] for d in sample["answers"]]
                #verify image_id is int 

                if not isinstance(sample["image_id"], int) :
                    print(sample["img_id"])
                    raise Exception("Image id not int")


                combined_sample = {
                    "question_id" : sample["question_id"] , 
                    "question" :question , 
                    "question_type" : sample["question_type"], 
                    "answer" : sample["multiple_choice_answer"],
                    "image_id" : sample["image_id"], #must be int 
                    "dataset" : "vqa"
                }
            

                combined_list.append(combined_sample)
        
        dir_path = os.path.dirname(ques_file)
        comb_path = os.path.join(dir_path, f"{split_name}_combined_ann.json")
        print(comb_path)
        with open(comb_path, 'w') as file : 
            json.dump(combined_list, file)


if __name__ == "__main__" : 
    file_list = [

        ("train", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_questions.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_annotations.json"), 
        ("val", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_questions.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_annotations.json"), 
        ("ID_test", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_questions.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_annotations.json"), 
        ("KO", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/OOD-Test-KO-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/OOD-Test-KO-Ans.json"),
        ("KOP", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/OOD-Test-KOP-Ques.json","/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/OOD-Test-KOP-Ans.json"), 
        ("KW", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/OOD-Test-KW-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/OOD-Test-KW-Ans.json"), 
        ("KW_KO", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/OOD-Test-KW+KO-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/OOD-Test-KW+KO-Ans.json"), 
        ("KWP", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/OOD-Test-KWP-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/OOD-Test-KWP-Ans.json"), 
        ("QT", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ans.json"), 
        ("QT_KO", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/OOD-Test-QT+KO-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/OOD-Test-QT+KO-Ans.json"), 
        ("QT_KW", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/OOD-Test-QT+KW-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/OOD-Test-QT+KW-Ans.json"), 
        ("QT_KW_KO", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/OOD-Test-QT+KW+KO-Ques.json", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/OOD-Test-QT+KW+KO-Ans.json"),
    ]


    org_full_ds(file_list)







