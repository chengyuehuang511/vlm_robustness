from transformers import AutoProcessor, PaliGemmaForConditionalGeneration,AutoImageProcessor, ViTModel

from PIL import Image
import requests
import torch 
import os
import json



f = "/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/res_file_path.json"

data = json.load(open(f))

for ft_method in data :
    print("Running for ", ft_method)
    for test_split in data[ft_method]: 
        print("TEST SPLIT", test_split)



# root_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result"

# for dir in os.listdir(root_dir):
#     next_level_dir = os.path.join(root_dir, dir)
    
#     if os.path.isdir(next_level_dir):  # Check if it's a directory
#         # Iterate over the files in the immediate subdirectory
#         for file_name in os.listdir(next_level_dir):
#             file_path = os.path.join(next_level_dir, file_name)
            
#             if os.path.isfile(file_path):  # Check if it's a file
#                 print(file_path)
        
                
#                 indiv_results = torch.load(file_path, map_location=torch.device('cpu'))

#                 for concept, instances in indiv_results.items():
#                     for instance_id, score in instances.items():
#                         indiv_results[concept][instance_id] = -score

#                 #verify 
#                 for concept, instances in indiv_results.items():
#                     for instance_id, score in instances.items():
#                         if indiv_results[concept][instance_id] < 0 :
#                             raise Exception("not positive maha score")
                        

#                 torch.save(indiv_results, file_path)
                

# #get the results dict for each ft_method and test_split
# root_dir = "/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA"

# ft_methods = ["lora", "digrap", "fft","ftp","lp","lpft","spd", "pt_emb"]

# test_splits = [
#     # "coco_vqav2_train_val",
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
# ]

# performance_file_path = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/perf_dict.json"

# with open(performance_file_path, 'r') as file : 
#     perf_dict = json.load(file)

# res_file_dict = {}
# res_file_path = os.path.join("/nethome/bmaneech3/flash/vlm_robustness/ood_test/other", "res_file_path.json")


# for ft_method in ft_methods : 
#     res_file_dict[ft_method] = {}
#     print(f"Running for {ft_method}")


#     for test_split in test_splits : 
#         print(f"Running for {test_split}")
#         score = perf_dict[ft_method][test_split]
        

#         res_path = None
#         for root, dirs, files in os.walk(root_dir):
#             if res_path != None : 
#                 break


#             for d in dirs : 
            

#                 if d.lower() in test_split or (test_split == "coco_vqa_cp_val" and d.lower()=="vqacp") or (test_split=="coco_vqa_raw_val" and d.lower() == "VQA") : 
#                     next_level_dir = os.path.join(root, d)

#                     # print(next_level_dir)

#                     for sub_root, sub_dirs, sub_files in os.walk(next_level_dir): 
#                         if "evaluate.txt" in sub_files:
#                             evaluate_path = os.path.join(sub_root, "evaluate.txt")

#                             with open(evaluate_path, 'r') as eval_file:
#                                 lines = eval_file.readlines()
#                                 last_entry = lines[-1].strip() 
#                                 metric = json.loads(last_entry)["agg_metrics"]

#                             if metric == score : 
#                                 res_path = os.path.join(sub_root, "result", "val_vqa_result.json")

#                                 break 

#         if res_path == None :
#             try :
#                 raise Exception(f"Path not found for {ft_method}, {test_split}")
#             except Exception as e : 
#                 print(e)

#         else :                   
        
#             res_file_dict[ft_method][test_split] = res_path




# with open(res_file_path, 'w') as file : 
#     json.dump(res_file_dict, file)



# f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/fft/coco_vqa_rephrasings_val_indiv_result.pth"
# t = torch.load(f, map_location=torch.device('cpu'))

# print(t.keys())




# model_id = "google/paligemma-3b-ft-vqav2-224"
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
# processor = AutoProcessor.from_pretrained(model_id)

# prompt = "What is on the flower?"

# image_file = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/images/0054c91397f2fe05.jpg"

# image = Image.open(image_file)

# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
# inputs = image_processor(image, return_tensors="pt")
# # print(inputs["attention_mask"])
# print(inputs.keys())
# with torch.no_grad():
#     outputs = model(**inputs, output_hidden_states=True)

# image_hidden_states = outputs.hidden_states[-1]
# print(image_hidden_states.size())

# image_emb = torch.mean(image_hidden_states, dim=1)


# print(image_emb.size())






# img_list = ["/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/images/0054c91397f2fe05.jpg", "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/images/c25292aeb1fbf1a3.jpg","/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/images/226d623d0c70664f.jpg","/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/images/c25292aeb1fbf1a3.jpg"]
# for image_file in img_list : 

#     raw_image = Image.open(image_file)
#     inputs = processor(images=raw_image, text=prompt, return_tensors="pt")
#     co = 0 
#     for i in inputs["input_ids"][0] : 
#         if i == 257152 : 
#             co += 1 

#     if co != 256 : 
#         raise Exception("not 256")
# print("sUCCESSFULLY ALL 256 TOKENS")

# print(inputs.keys())

# print(inputs["attention_mask"].size()) #(batch size, seq length)
# attention_mask = inputs["attention_mask"]
# attention_mask[:, :256] = 0
# """dict_keys(['input_ids', 'attention_mask', 'pixel_values'])"""

# print(attention_mask.size())
# print(inputs["input_ids"][0][:256])


# print("question token length :", len(inputs["input_ids"][0][256:]))
# print(f"question length : {len(prompt)}")




# print("input-ids: \n", inputs["input_ids"][:256])
# with torch.no_grad() : 
#     output = model(**inputs,  return_dict=True, output_hidden_states=True, output_attentions=True)

# for i in range(len(raw_image)):
#     attn = output.attentions[-1][i].mean(dim=0)

#     img_token_idx = inputs['input_ids'][i] == 257152
#     print(img_token_idx.size())








# print(output.hidden_states.size())

# image_portion = output.hidden_states[-1][:, :256, :] 

# ques_portion = output.hidden_states[0][:,256:, :]







# if (image_portion == 0).all() : 
#     print("yayyyy")
# print(image_portion[0])

# # print(processor.decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ])


# import torch 

# # a = torch.tensor([1,2,3,4])

# # print(a[:len(a)])

# a = torch.full((2048, 4), 3)
# cov_matrix = torch.cov(a)
# print(cov_matrix.size())



# # f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/fft/coco_cv-vqa_val_image_joint.pth"
# # data = {int(key) : value  for key, value in torch.load(f).items()}
# # for idx, (key, value) in enumerate(sorted(data.items())): 
# #     if idx != key : 
# #         raise Exception("not match")

# # print("Succesful")


# t = torch.zeros(2)
# arr = torch.tensor([1,2])
# t = arr + t 
# # .permute(1,0).squeeze(1)
# # print(arr.size())
# # print(t.size())
# print( t)