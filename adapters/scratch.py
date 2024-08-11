from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from PIL import Image
import os 
import torch 
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchvision import transforms
from torchvision.transforms import InterpolationMode


import sys 

sys.path.append('/nethome/bmaneech3/flash')
# sys.path.append('/nethome/bmaneech3/flash/LLM-Adapters/peft/src/peft')
# sys.path.append('/nethome/bmaneech3/flash/LLM-Adapters')

from llm_adapters.peft.src.peft import (  # noqa: E402
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)




from llm_adapters.peft.src.peft.tuners.bottleneck import BottleneckConfig


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


print("Paligemma module inspection")

# print(model.name)
# key_list = [key for key, _ in model.named_modules()]
# for key in key_list:
#     target = model.get_submodule(key)
#     if isinstance(target, torch.nn.Linear):
#         print("YESS",key.split(".")[-1])
        


# print(model.config.model_type)

# config = BottleneckConfig(
#             bottleneck_size=256,
#             non_linearity="tanh",
#             adapter_dropout=0.0,
#             use_parallel_adapter=False,
#             use_adapterp=False,
#             target_modules=['q_proj','v_proj','k_proj','o_proj'],
#             scaling=1.0,
#             bias="none",
#             task_type="CAUSAL_LM",
# )


# key_list = [key for key, _ in model.named_modules()]
# for key in key_list:
#     target = model.get_submodule(key)
#     print(target)
#     if target.requires_grad :
#         print(key, target) 

    # if isinstance(target, torch.nn.Linear):
    #     print("YESS",key.split(".")[-1])
        



config = PrefixTuningConfig()
# print(model)
print("CONFIGGG", model.config)
new_model = get_peft_model(model, config)

# print(model)
# print("trainable parameters")
# model.print_trainable_parameters()
# for layer_name, params in model.named_modules():
#     if params.requires_grad: 
#         print(f"Layer: {layer_name}, Size: {params.size()}")

# for layer_name, param in model.named_parameters():
#     if param.requires_grad: 
#         print(f"Layer: {layer_name}, Size: {param.size()}")



print("FINALL FINISHED MODEL")
print(new_model)

