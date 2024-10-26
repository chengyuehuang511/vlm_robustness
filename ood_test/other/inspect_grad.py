import json
f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/ans_map.json"
with open(f, 'r') as file : 
    data = json.load(file)
    print(len(data))
    print(data['net'])
import sys 
from pathlib import Path
import torch


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from model.paligemma_vqa import PaliGemma_VQA 
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration




ckpt_path = "/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/VQA/spcg_1e-3_mu0.5/20240918154/checkpoint_best.pth"

# model_id = "google/paligemma-3b-ft-vqav2-224"
# model = PaliGemma_VQA(model_id, device_map ="auto",
# torch_dtype=torch.bfloat16,
# revision="bfloat16").from_pretrained(ckpt_path)

pt_model_id = "google/paligemma-3b-pt-224"
ft_model_id = "google/paligemma-3b-ft-vqav2-224"

pt_model = PaliGemmaForConditionalGeneration.from_pretrained(pt_model_id, 
                                                        device_map ="auto",
                                                        torch_dtype=torch.bfloat16,
                                                        revision="bfloat16").eval()

ft_model = PaliGemmaForConditionalGeneration.from_pretrained(ft_model_id, 
                                                        device_map ="auto",
                                                        torch_dtype=torch.bfloat16,
                                                        revision="bfloat16").eval()


# ckpt_path = "/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/VQA/spcg_1e-3_mu0.5/20240918154/checkpoint_best.pth"

# ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

# #don't forget to put model to device 
# model_id = "google/paligemma-3b-pt-224"
# pt_model = PaliGemma_VQA(model_id,
# dtype=torch.bfloat16,
# ).from_config(ckpt['config']['model'])
# # pt_model.load_from_pretrained(ckpt_path)
# pt_model.eval()
# # print("MODEL **", model.base_model.model.model)

# # pt_model.base_model.model.model.device_map = "auto"
# pt_model = pt_model.base_model.model.model


# new_config = ckpt['config']['model']
# new_config["load_finetuned"] = True 

# # model = model.to(device)

# ft_model = PaliGemma_VQA(model_id,
# dtype=torch.bfloat16,
# ).from_config(new_config)
# # pt_model.load_from_pretrained(ckpt_path)
# ft_model.eval()
# ft_model = ft_model.base_model.model.model


# Dictionary to store the differences
weight_differences = {}

# Iterate over the parameters of both models
for (name_pretrained, param_pretrained), (name_fine_tuned, param_fine_tuned) in zip(pt_model.named_parameters(), ft_model.named_parameters()):
    
    # Ensure the parameter names are the same
    assert name_pretrained == name_fine_tuned, f"Mismatch in parameter names: {name_pretrained} vs {name_fine_tuned}"
    
    # Compute the difference between pre-trained and fine-tuned weights
    diff = torch.abs(param_pretrained - param_fine_tuned)
    print(diff.size())

    non_zero_diff = diff[diff != 0]
    #drop zeros before the mean
    
    if non_zero_diff.numel() > 0:  # Check if there are non-zero elements
        weight_differences[name_pretrained] = torch.mean(non_zero_diff).item()
    else:
        weight_differences[name_pretrained] = 0  # Assign zero if no non-zero elements

    
    weight_differences[name_pretrained] = torch.mean(diff).item()
sorted_differences = sorted(weight_differences.items(), key=lambda x: x[1], reverse=True)
# Print out the most updated layers (you can adjust the number of layers to show)
print("Top 10 most updated layers after fine-tuning:")
for layer, diff in sorted_differences:  # Top 10 most updated layers
    print(f"Layer: {layer}, Mean Difference: {diff}")


print("max update")
top_layer, top_diff = sorted_differences[0]
bottom_layer, bottom_diff = sorted_differences[-1]

print(f"Max Layer: {top_layer}, Max Mean Difference: {top_diff}")
print(f"Min Layer: {bottom_layer}, Min Mean Difference: {bottom_diff}")