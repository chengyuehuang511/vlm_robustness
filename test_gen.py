from lavis.models import load_model_and_preprocess
import torch 
from PIL import Image 
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)

# text = "What is the person making?"
# "image_id": 130240, "question": "What hand is the man catching with?"

question = "What hand is the man catching with?"

raw_image = Image.open("/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000130240.jpg").convert("RGB")


image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
question = txt_processors["eval"](question)


print(model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate"))






# with open('/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/combined_data.json', 'r') as file : 
#     data = json.load(file)
    
#     for sample in data : 

#         if int(sample["question_id"]) == 130240000 : 
#             print(sample["image"])













