import torch 
from collections import defaultdict
f = "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/v2_OpenEnded_mscoco_val2014_questions.json"
import json
# combine_path = "/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/ds_split_2_file.json"
# with open(combine_path, 'r') as file : 
#     ds_split_2_file = json.load(file)
#     test_splits = [
#     "coco_vqav2_train_val",
#     "coco_advqa_val", 
#     "coco_cv-vqa_val",
#     "coco_iv-vqa_val",
#     "coco_okvqa_val",
#     "coco_vqa_ce_val",
#     "coco_vqa_cp_val",
#     "coco_vqa_raw_val",
#     "coco_vqa_rephrasings_val",
#     "textvqa_val",
#     "vizwiz_val"
#     ]

#     for split, f in ds_split_2_file.items() : 
#         if "vqa_vs" in split or "vqa_lol" in split or "vqa_v2_test" in split: 
#             continue 
    
        
#         print("Split", split)
#         # file_name = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/ft_indiv_result/question/{split}_indiv_result.pth"
#         # # with open(file_name, 'r') as file : 
#         # data = torch.load(file_name,  map_location=torch.device('cpu'))
#         # print("len", len(data["question"]))
#         # indiv_len = len(data)

#         with open(f, 'r') as file : 
#             data = json.load(file)
#             print(len(data))

        # print(len(data) - indiv_len)
        # print(f"{(len(data) - indiv_len)/len(data)} %")



            # ques_dict  = defaultdict(list)
            # print(data[0].keys())
            # print(data[0]["question_id"])
            # if "image_id" in data[0] : 
            #     print(data[0]["image_id"])
            # if "image" in data[0] : 
            #     print(data[0]["image"])

        # print()
    # co = 0 
    # for sample in data : 
    #     if sample["question_id"] in ques_dict and sample["image_id"] not in ques_dict[sample["question_id"]]: 
    #         # print("yess") 
    #         co += 1 
        
    #     ques_dict[sample["question_id"]].append(sample["image_id"])
    

    # print(len(ques_dict))

    # print(co)

# print(d.size())

# f =  "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json"
# with open(f, 'r') as file : 
#     data = json.load(file)
#     ques_dict  = defaultdict(list)
#     print(data[0].keys())
#     print(data[0]["question_id"])
#     if "image_id" in data[0] : 
#         print(data[0]["image_id"])
#     if "image" in data[0] : 
#         print(data[0]["image"])



# f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/ft_indiv_result/digrap/coco_advqa_val_indiv_result.pth"
# f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/lora/coco_okvqa_val_image_joint.pth"
# data = torch.load(f)
# # print(data[str(0)].size())
# # print(data.size())
# # data = torch.load(f, map_location=torch.device('cpu'))
# # print(data.keys())
# # for i in range(100) : 
# #     print(data["joint"][i]) #()

# f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/pt_hidden_states/advqa_test_pretrain_img_joint.pth"
# data = torch.load(f)
# # print(data.keys())
# print(data.size())



print("Starts here")
import torch
import os
root_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states"
new_root_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_hidden_states"

ft_methods = ["digrap", "fft", "ftp", "lora", "lp", "lpft", "pt_emb", "spd", "pt_emb", "vit"]
os.makedirs(new_root_dir, exist_ok=True)


for ft_method in ft_methods:
    print(f"========== current FT Method {ft_method}")
    method_dir = os.path.join(root_dir, ft_method)
    new_method_dir = os.path.join(new_root_dir, ft_method)
    os.makedirs(new_method_dir, exist_ok=True)
    
    # Traverse through files in each method directory
    for root, _, files in os.walk(method_dir):
        for file in files:
            #image-joint files 
            if file.endswith("image_joint_new.pth"):
                file_path = os.path.join(method_dir, file)
                print(f"Processing file : {file_path}")

                t = torch.load(file_path) # instance_id : (concept, dim)

                image_emb = {key: value[0,:] for key, value in t.items()}


                # image_emb = t[:, 0, :] 
                try : 
                    assert len(image_emb) == len(t) and image_emb['0'].size() == torch.Size([2048]), "image concept embs incorrect extraction"

                except AssertionError as e : 
                    print(e)

                
                joint_emb  = {key: value[1,:] for key, value in t.items()}
                try : 
                    assert len(joint_emb) == len(t) and joint_emb['0'].size() == torch.Size([2048]), "joint concept embs incorrect extraction"

                except AssertionError as e : 
                    print(e)

                new_img_file_name = file.replace("image_joint_new.pth", "image_ft.pth")
                image_store_file = os.path.join(new_method_dir, new_img_file_name)

                new_joint_file_name = file.replace("image_joint_new.pth", "joint.pth")
                joint_store_file = os.path.join(new_method_dir, new_joint_file_name)

                torch.save(image_emb, image_store_file)
                torch.save(joint_emb, joint_store_file) 

            if file.endswith("ques_ft_new.pth"):
                
                file_path = os.path.join(method_dir, file)
                print(f"Processing file : {file_path}")
                t = torch.load(file_path) #(batch size, concept, hidden dim)
                print(type(t))

                ques_ft_emb = {key: value[0,:] for key, value in t.items()}
                try : 
                    assert len(ques_ft_emb) == len(t) and ques_ft_emb['0'].size() == torch.Size([2048]), "ques ft concept embs incorrect extraction"

                except AssertionError as e : 
                    print(e)

                new_ques_ft_file_name = file.replace("ques_ft_new.pth", "ques_ft.pth")
                ques_ft_store_file = os.path.join(new_method_dir, new_ques_ft_file_name)

                torch.save(ques_ft_emb, ques_ft_store_file)

            if file.endswith("question_new.pth") : 
                file_path = os.path.join(method_dir, file)
                print(f"Processing file : {file_path}")
                t = torch.load(file_path) #(batch size, concept, hidden dim)

                question_emb = {key: value[0,:] for key, value in t.items()}
                try : 
                    assert len(question_emb) == len(t) and question_emb['0'].size() == torch.Size([2048]), "question concept embs incorrect extraction"

                except AssertionError as e : 
                    print(e)

                new_question_file_name = file.replace("question_new.pth", "question.pth")
                os.makedirs("/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_hidden_states/bert", exist_ok=True)
                question_store_file = os.path.join("/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_hidden_states/bert", new_question_file_name)

                torch.save(question_emb, question_store_file)


            if file.endswith("uni_image_new.pth") : 
                file_path = os.path.join(method_dir, file)
                print(f"Processing file : {file_path}")
                t = torch.load(file_path) #(batch size, concept, hidden dim)

                image_emb = {key: value[0,:] for key, value in t.items()}
                try : 
                    assert len(image_emb) == len(t) and image_emb['0'].size() == torch.Size([2048]), "image concept embs incorrect extraction"

                except AssertionError as e : 
                    print(e)

                new_image_file_name = file.replace("uni_image_new.pth", "image.pth")
                image_store_file = os.path.join(new_method_dir, new_image_file_name)

                torch.save(image_emb, image_store_file) 


            










                












