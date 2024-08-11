from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from PIL import Image
import os 
import torch 
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# from data.builders import load_dataset, DatasetZoo

# dataset_zoo = DatasetZoo()
# # coco_vqa_vs
# vqa_vs = load_dataset("coco_vqa_vs")

image = 448

model_id = "google/paligemma-3b-ft-vqav2-224"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")
processor = PaliGemmaProcessor.from_pretrained(model_id)

prompt = "What is in the image right now"
image_file = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/images/226d623d0c70664f.jpg"
raw_image = Image.open(image_file).convert("RGB")
resize_transform = transforms.Resize(
    (448, 448), interpolation=InterpolationMode.BICUBIC
)
raw_image = resize_transform(raw_image)

#split into image and prompt
inputs = processor(prompt, raw_image, return_tensors="pt")
# print(inputs.keys())
# print(inputs["input_ids"].size())
# print(inputs["attention_mask"].size())



# key: torch.cat([item[0][key] for item in batch], dim=0)
# for key in batch[0][0].keys()

output = model.forward(**inputs, return_dict=True, output_hidden_states=True)
# print(output)
hidden_states = output.hidden_states
# print(hidden_states)
# print(type(hidden_states))
# print(hidden_states)
# print(output)
# for layer_name, params in model.named_modules():
#     print(layer_name)

# print(len(hidden_states))
# output_hidden_state = hidden_states[-1] #average all vectors together  
# image_hidden_state = hidden_states[0][:, 0:256, :] #first image encoder after projection 

# """
# output_hidden_state : (batch size, seq length, hidden dim)
# image_hidden_state : (batch size, seq length, hidden dim)
# """

# u_image_vector = torch.mean(image_hidden_state, dim=1)
# u_output_hidden_vector = torch.mean(output_hidden_state, dim=1)

# print(u_output_hidden_vector.size())
# print(u_image_vector.size())




#try scratch 

key_list = [key for key, _ in model.named_modules()]
for key in key_list:
    # target = model.get_submodule(key)
    print(key)
    
    # if isinstance(target, torch.nn.Linear):
    #     print("YESS",key.split(".")[-1])





