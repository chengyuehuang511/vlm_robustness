import os
import json 
#VQA_VS
f = [
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_annotations.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_annotations.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_annotations.json"
] 



questions = { 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_questions.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/train_questions.json" , 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_questions.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/val_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_questions.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/test_questions.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/OOD-Test-KO-Ques.json": "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/test_questions.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/OOD-Test-KOP-Ques.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/test_questions.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/OOD-Test-KW-Ques.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/OOD-Test-KW+KO-Ques.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/OOD-Test-KWP-Ques.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ques.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/OOD-Test-QT+KO-Ques.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/OOD-Test-QT+KW-Ques.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/OOD-Test-QT+KW+KO-Ques.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/test_questions.json"


}
annotations = {
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_annotations.json": "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/train_annotations.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_annotations.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/val_annotations.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_annotations.json": "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/test_annotations.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/OOD-Test-KO-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/test_annotations.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/OOD-Test-KOP-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/test_annotations.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/OOD-Test-KW-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/test_annotations.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/OOD-Test-KW+KO-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/test_annotations.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/OOD-Test-KWP-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/test_annotations.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/test_annotations.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/OOD-Test-QT+KO-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/test_annotations.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/OOD-Test-QT+KW-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/test_annotations.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/OOD-Test-QT+KW+KO-Ans.json" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/test_annotations.json"


}

tuple_data = [ 
    ("/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_annotations.json"), 
    ("/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/vqa_vs_val_annotations.json"), 
    ("/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_annotations.json")
]


#map image id to corresponding image 

image_dir = "/coc/pskynet6/chuang475/.cache/lavis/coco/images"


split = ["test2014" ,"test2015",  "train2014",  "val2014"] 
duplicate_list = [] 
def extract_image_to_id() : 

    id_2_split = {} 

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            
            if file.endswith(".jpg") : 
                # print(file)

                id = file.split("_")[2][:-4]
                split = file.split("_")[1]
                id = int(id)

                image = f"{split}/{file}"

                # if (id == 15089) :
                #     print("15089", split)

                if id in id_2_split:  
                    if ("train" in split and not "train" in id_2_split[id]) or (not "train" in split and "train" in id_2_split[id]) : 
                        raise Exception("key duplicate id already exists")

                    continue
                    
                    
                    # print("split: ", split)
                    # print("image id : ", id)

                    # print("split: ", id_2_split[id])
                    # print("image id : ", id)
                    # duplicate_list.append((id_2_split[id], image))
                id_2_split[id] = image

    return id_2_split

    #iterate through all the directories 



"""check duplicate image"""
# from PIL import Image
# co = 0

#     for first, sec in duplicate_list : 

#         img = Image.open(f"/coc/pskynet6/chuang475/.cache/lavis/coco/images/{first}")
#         img.save(f"/nethome/bmaneech3/flash/vlm_robustness/organize_dataset/img{co}.jpg")
#         img = Image.open(f"/coc/pskynet6/chuang475/.cache/lavis/coco/images/{sec}")
#         img.save(f"/nethome/bmaneech3/flash/vlm_robustness/organize_dataset/dupimg{co}.jpg")
#         co+=1 





if __name__ == "__main__" : 
    id_2_split = extract_image_to_id()


    quesid_2_ques = {} 
    #other checks : unique question id 

    for ques_file in questions : 
        question_dict = {
            "info" : None,
            "task_type" : "Open-Ended",
            "data_type": "vqavs",
            "data_subtype": None,
            "questions" : [],
            "license" : None,
            # "num_choices": int 
        }

        with open(ques_file, 'r') as file : 

            #find image_id 

            data = json.load(file)

            
            for sample in data : 
                image = id_2_split[sample["image_id"]]
                split = image.split("_")[1]

                ques_sample = {
                    "coco_split" : split,  
                    "image_id" : sample["image_id"], 
                    "question" : sample["question"], 
                    "question_id" : sample["question_id"] 
                } 

                # if sample["question_id"] in quesid_2_ques : 
                    # "duplicate question id error")
                
                quesid_2_ques[sample["question_id"]] = sample["question"]
                question_dict["questions"].append(ques_sample)


        desti_path = questions[ques_file]  
        with open(desti_path,'w') as file : 
            json.dump(question_dict, file)



    #process annotation and combined file 

    for ann_file in annotations : 
        ann_dict = {
        "info" : None ,
        "data_type": "vqavs",
        "data_subtype": None,
        "annotations" : [],
        "license" : None
        }

        combined_list = []  

        with open(ann_file, 'r') as file : 

            data = json.load(file)


            for sample in data : 

                ann_dict["annotations"].append(sample)

                ques_id = sample["question_id"]
                question = quesid_2_ques[ques_id]

                answer_list = [d["answer"] for d in sample["answers"]]

                combined_sample = {
                    "question_id" : sample["question_id"] , 
                    "question" :question , 
                    "answer" : answer_list, 
                    "image": id_2_split[sample["image_id"]] , 
                    "dataset" : "vqa"
                }
            

                combined_list.append(combined_sample)

        
        desti_path = annotations[ann_file]  
        with open(desti_path,'w') as file : 
            json.dump(ann_dict, file)

        
        dir_path = os.path.dirname(desti_path)
        comb_path = os.path.join(dir_path, "combined_data.json")
        print(comb_path)
        with open(comb_path, 'w') as file : 
            json.dump(combined_list, file)





