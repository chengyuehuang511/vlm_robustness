import json 


def display_dataset(file_name, file_type) : 

    with open(file_name,'r') as file : 
        data = json.load(file)
        print("Dataset type", type(data))
    

        if isinstance(data, dict) : 
            print("DICT KEYS", data.keys())
            # print("task type", data["task_type"])
            # print("data type", data["data_type"])
            # print("sub type", data["data_subtype"])
            print("info", data["info"])
            print()
            data = data[file_type] 
            
        print("Dataset keys: ", data[0].keys())

        print("total samples ", len(data))
        try : 
            print("image id: ", data[0]["image_id"])

        except : 
            try : 
                print("image id: ", data[0]["img_id"])

            except : 
                print("No image id found")


        if  "image" in data[0] : 
            print("Image: ", data[0]["image"])


        if "answer" in data[0] : 
            print(data[1000]["answer"])

        if "answers" in data[0] : 
            print(data[1000]["answers"])

        if "question_type" in data[0] :
            print("question type",data[1000]["question_type"])
            
f = [ 

    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/ID_test_combined_ann.json"
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json",
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json"
    # "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/v1_mscoco_val2017_advqa_annotations_new.json"

    # "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/v2_mscoco_val2014_annotations.json"
    "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/combined_data.json"
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_eval.json"
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json"
]

print("VQA LOL structure")
for i in f : 
    print(f"File name {i}")
    if "question" in i or "questions" in i or "combined_data" in i : 
        print(display_dataset(i,"questions"))

    else : 
        print(display_dataset(i, "annotations"))



# f = "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json"
# with open(f, 'r') as file : 
#     data = json.load(file)
#     print(len(data))












# ds_split_2_file = { 
#     "vqa_v2_train" : "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json",
#     "vqa_v2_val" : "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_eval.json", 
#     "vqa_v2_test": "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json" , 
#     "advqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json", 
#     "cvvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/combined_data.json", 
#     "ivvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/combined_data.json",
#     "okvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/ok-vqa/val/combined_data.json",
#     "textvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/combined_data.json",
#     "textvqa_train" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/train/combined_data.json", 
#     "vizwiz_test" :  "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/combined_data.json",
#     "vqa_ce_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/combined_data_subset.json",
#     "vqa_cp_train" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_questions.json", 
#     "vqa_cp_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/combined_data.json", 
#     "vqa_lol_train": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/train/combined_data.json",
#     "vqa_lol_test": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/test/combined_data.json", 
#     "vqa_rephrasings_test": "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqa_rephrasings/combined_data.json",
#     "vqa_vs_train" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/combined_data.json", 
#     "vqa_vs_id_val" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/combined_data.json", 
#     "vqa_vs_id_test" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/ID_test_combined_ann.json", 
#     "vqa_vs_ood_test" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
#     "vqa_vs_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/combined_data.json",
#     "vqa_vs_KOP" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/combined_data.json", 
#     "vqa_vs_KW" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/combined_data.json", 
#     "vqa_vs_KW_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/combined_data.json", 
#     "vqa_vs_KWP" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/combined_data.json", 
#     "vqa_vs_QT" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/combined_data.json", 
#     "vqa_vs_QT_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/combined_data.json", 
#     "vqa_vs_QT_KW" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/combined_data.json", 
#     "vqa_vs_QT_KW_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/combined_data.json"
# }




# for ds, file in ds_split_2_file.items() :

#     with open(file, 'r') as f : 
#         data = json.load(f)

#         sample = data[0]

#         if "question" not in sample or "answer" not in sample or "image" not in sample : 
#             print(f"{ds} incorrect, {sample.keys()}")














