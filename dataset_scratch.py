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
    "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json"
    # "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json"
    # "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/v1_mscoco_val2017_advqa_annotations_new.json"

    # "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/v2_mscoco_val2014_annotations.json"
    # "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json"
]

# print("VQA LOL structure")
for i in f : 
    print(f"File name {i}")
    if "question" in i or "questions" in i or "combined_data" in i : 
        print(display_dataset(i,"questions"))

    else : 
        print(display_dataset(i, "annotations"))














