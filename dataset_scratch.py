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



        # print(data[1000]["question_family_index"])



        # for answer in 

f = [ 
    # "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json"
    
    # "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_questions.json", 
    # "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_annotations.json", 
    # "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/combined_data.json", 
    # "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/vqacp_v2_test_questions.json",
    # "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/vqacp_v2_test_annotations.json",

    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json", 
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_eval.json",
    # "/srv/kira-lab/share4/schopra47/VQA/data/vqa_v2/v2_mscoco_train2014_annotations.json"
    
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json"
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val.json",
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json",
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_mscoco_val2014_annotations.json"

    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqalol/test/nominival_vqa_lol_questions.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqalol/test/nominival_vqa_lol_annotations.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqalol/test/combined_data.json",

    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqalol/train/combined_data.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqalol/train/train_vqa_lol_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqalol/train/train_vqa_lol_annotations.json"

    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json"
    # "/srv/kira-lab/share4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2_new/raw/annotations/mscoco_train2014_annotations.json"
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/train_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/train_annotations.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/val_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/val_annotations.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/test_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/test_annotations.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json" 

    # "/srv/kira-lab/share4/schopra47/VQA/data/clevr/CLEVR_v1.0/questions/CLEVR_train_questions.json"
]


"""
inspecting VQA VS 
"""


f = [ 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/vqa_vs_iid_test_annotations.json"
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/OOD-Test-QT-Ans.json"
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/vqa_vs_train_annotations.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/val_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/val_annotations.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/test_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/test_annotations.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/combined_data.json"

    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/OOD-Test-KW+KO-Ans.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/OOD-Test-KW+KO-Ques.json"
    

]

# print("VQA LOL structure")
for i in f : 
    print(f"File name {i}")
    if "questions" in i or "combined_data" in i : 
        print(display_dataset(i,"questions"))

    else : 
        print(display_dataset(i, "annotations"))

#DICT KEYS dict_keys(['info', 'task_type', 'data_type', 'license', 'data_subtype', 'questions'])
#DICT KEYS dict_keys(['info', 'license', 'data_subtype', 'annotations', 'data_type'])

#task type Open-Ended
#data type mscoco
#sub type val2014

