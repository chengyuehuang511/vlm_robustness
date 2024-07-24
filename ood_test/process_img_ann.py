

import json 
import os
from tqdm import tqdm
dir_root = "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/sample_org"

def create_image_index(file_list) : 

    img_id_samples = {}  #map img id -> everything (file_name, category list, url_list)
    cat_id_samples = {} 


    #merge all image splits together -> verify that they are same 

    for file_name in file_list:
        with open(file_name, 'r') as file : 
            print("hello")

            data = json.load(file)

            anns = data["annotations"] #list of dict 
            imgs = data["images"] 
            categories = data["categories"]

            id_2_cat = {} 
            id_2_images = {} 
            id_2_ann = {} 
            

            for cat in tqdm(categories, desc=f"Processing categories in {file_name}", leave=False):
                id = cat["id"] 
                id_2_cat[id] = {
                    "name" : cat["name"], 
                    "supercategory" : cat["supercategory"]
                }

            for img in imgs:
                id = img["id"]
                # print(id)
                # id_2_images[id] = { 
                #     "file_name" : img["file_name"], 
                #     "coco_url" : img["coco_url"],
                #     "categories" : [] 
                # }
                # if id == 262184 :
                #     print("found \n\n")
            
            for ann in anns:
                #ann = dict 
                img_id = ann["image_id"] 
                cat_id = ann["category_id"]

                if img_id not in id_2_images : 
                    id_2_images[img_id] = { 
                    "file_name" : img["file_name"], 
                    "coco_url" : img["coco_url"],
                    "categories" : [] 
                    }


                id_2_images[img_id]["categories"].append(cat_id)
                if img_id == 262184: 
                    print("cat_id", cat_id)
                    print(type(img_id))


            if 262184 not in id_2_images : 
                print("but not in anns")

            else : 
                print("found in annotations")
                print(id_2_images[262184]["categories"])

            # #check duplicates  

            for img_id in tqdm(id_2_images, desc="Checking duplicate images", leave=False):
                if img_id in img_id_samples : 
                    list_a = set(id_2_images[img_id]["categories"])
                    list_b = set(img_id_samples[img_id]["categories"])

                    if list_a != list_b : 
                        print(f"img id: {img_id }, list a : {list_a}, list b : {list_b}")
                        raise Exception("Duplicate error with img id")
                    


            for cat_id in tqdm(id_2_cat, desc="Checking duplicate categories", leave=False):
                if cat_id in cat_id_samples : 
                    list_a = id_2_cat[cat_id]["name"]
                    list_b = cat_id_samples[cat_id]["name"]

                    if list_a != list_b : 
                        print(f"cat id: {cat_id }, name a : {list_a}, name b : {list_b}")

                        raise Exception("Duplicate error with cat id")


            img_id_samples.update(id_2_images)
            cat_id_samples.update(id_2_cat) 


    print("Final counts")
    print(f"total images: {len(img_id_samples)}")
    print(f"total categories: {len(cat_id_samples)}")

    #img dict file 
    img_file_path = os.path.join(dir_root, "trainval_img_samples.json")
    with open(img_file_path, 'w') as file : 
        json.dump(img_id_samples, file)

    cat_file_path = os.path.join(dir_root, "trainval_keyobj_samples.json")
    with open(cat_file_path, 'w') as file : 
        json.dump(cat_id_samples, file)





if __name__ == "__main__"  : 
    file_list = [
        "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/coco/annotations/annotations/instances_train2014.json", 
        "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/coco/annotations/annotations/instances_val2014.json"
        # "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/coco/annotations/annotations/captions_train2014.json"
    ]
    create_image_index(file_list)





#double check the img id problem 




























