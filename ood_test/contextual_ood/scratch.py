from transformers import AutoProcessor, PaliGemmaForConditionalGeneration,AutoImageProcessor, ViTModel
from PIL import Image
import requests
import torch 
model_id = "google/paligemma-3b-ft-vqav2-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "What is on the flower?"

image_file = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/images/0054c91397f2fe05.jpg"

image = Image.open(image_file)

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
inputs = image_processor(image, return_tensors="pt")
# print(inputs["attention_mask"])
print(inputs.keys())
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

image_hidden_states = outputs.hidden_states[-1]
print(image_hidden_states.size())

image_emb = torch.mean(image_hidden_states, dim=1)


print(image_emb.size())






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
with torch.no_grad() : 
    output = model(**inputs,  return_dict=True, output_hidden_states=True, output_attentions=True)

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