import sys
import os

# Add the directory containing `model` to the path
sys.path.append(os.path.abspath('/coc/testnvme/chuang475/projects/vlm_robustness/'))

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from model.paligemma_vqa import PaliGemma_VQA
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import gc 
import math 
import pandas as pd
from tqdm import tqdm

torch.cuda.empty_cache()

#globally set no gradient tracking
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds_2_img = { 
    "advqa" : "/srv/datasets/coco/", 
    "cvvqa" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/IMAGES_counting_del1_edited_VQA_v2/",
    "vqa_v2" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "ivvqa": "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/final_edited_VQA_v2/Images/", 
    "okvqa" : "/srv/datasets/coco/",
    "textvqa" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/", 
    "vizwiz" : "/srv/datasets/vizwiz/data/Images/", 
    "vqa_ce" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_cp" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_lol" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_rephrasings" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/",
    "vqa_vs" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/"
}

ds_split_2_file = { 
    "vqa_v2_train" : "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json",
    "vqa_v2_val" : "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_eval.json", 
    "vqa_v2_test": "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json" , 
    "advqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json", 
    "cvvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/combined_data.json", 
    "ivvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/combined_data.json",
    "okvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/ok-vqa/val/combined_data.json",
    "textvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/combined_data.json",
    "textvqa_train" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/textvqa/train/combined_data.json", 
    "vizwiz_test" :  "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/combined_data.json",
    "vqa_ce_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/combined_data_subset.json",
    "vqa_cp_train" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_questions.json", 
    "vqa_cp_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/combined_data.json", 
    "vqa_lol_train": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/train/combined_data.json",
    "vqa_lol_test": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/test/combined_data.json", 
    "vqa_rephrasings_test": "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqa_rephrasings/combined_data.json",
    "vqa_vs_train" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/combined_data.json", 
    "vqa_vs_id_val" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/combined_data.json", 
    "vqa_vs_id_test" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
    "vqa_vs_ood_test" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
    "vqa_vs_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/combined_data.json",
    "vqa_vs_KOP" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/combined_data.json", 
    "vqa_vs_KW" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/combined_data.json", 
    "vqa_vs_KW_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/combined_data.json", 
    "vqa_vs_KWP" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/combined_data.json", 
    "vqa_vs_QT" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/combined_data.json", 
    "vqa_vs_QT_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/combined_data.json", 
    "vqa_vs_QT_KW" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/combined_data.json", 
    "vqa_vs_QT_KW_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/combined_data.json"
}

ans_label_map = {} 
#store ans_label_map in json 


#don't forget to put model to device 
# model_id = "google/paligemma-3b-ft-vqav2-224"
model_id = "google/paligemma-3b-pt-224"
# print("model_id", model_id)

# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, 
#                                                         device_map ="auto",
#                                                         torch_dtype=torch.bfloat16,
#                                                         revision="bfloat16").eval()

model = PaliGemma_VQA(model_id=model_id, dtype=torch.bfloat16,).to(device).eval()

BATCH_SIZE = 4


class MeasureAttnDataset(Dataset) : 
    
    def __init__(self, data, ds_name):
        self.data = data #list of dictionary elements 
        self.ds_name = ds_name 
        self.vis_root = ds_2_img[ds_name]

        for idx, ann in enumerate(self.data):
            ann["instance_id"] = str(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]
        image_path = os.path.join(self.vis_root, ann["image"])
        image_raw = Image.open(image_path).convert("RGB")

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        # select the most frequent multiple_choice_answer in the list - ann["answer"]
        multiple_choice_answer = max(set(ann["answer"]), key=ann["answer"].count)

        return {
            "answers": answers,
            "multiple_choice_answer": multiple_choice_answer,
            "weights": weights,
            "image_raw": image_raw,
            "text_input_raw": ann["question"],
            "image_path": image_path,
            "instance_id": ann["instance_id"],
        }
    
    def collate_fn(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return None
        answer_list, weight_list = [], []
        image_raw_list, question_raw_list, multiple_choice_answer_list = [], [], []

        num_answers = []
        image_path_list = []
        instance_id_list = []

        for sample in samples:
            image_raw_list.append(sample["image_raw"])
            question_raw_list.append(sample["text_input_raw"])
            image_path_list.append(sample["image_path"])

            multiple_choice_answer_list.append(sample["multiple_choice_answer"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

            instance_id_list.append(sample["instance_id"])

        return {
            "image_raw": image_raw_list,
            "image_path": image_path_list,
            "text_input_raw": question_raw_list,
            "answer": answer_list,
            "weight": weight_list,
            "n_answers": torch.LongTensor(num_answers),
            "multiple_choice_answer": multiple_choice_answer_list,
            "instance_id": instance_id_list,
        }
    

"""
train ds name, split 
test ds name, split

concept : joint/image/question etc. 

"""


splits =[
    #ds_name, split, file_name
    #(train, test, concept)
    #vqav2 train with all others
    #train_stuff = sample[0], test_stuff = sample[1]
    # [("vqa_v2","train"), ("vqa_v2","train")],

    [("vqa_v2","train"), ("vqa_v2","val")], 
    [("vqa_v2","train"), ("advqa", "test")],
    [("vqa_v2","train"), ("cvvqa", "test")], 
    [("vqa_v2","train"), ("ivvqa", "test")],
    [("vqa_v2","train"), ("okvqa", "test")], 
    [("vqa_v2","train"), ("textvqa", "test")], 
    [("vqa_v2","train"), ("vizwiz", "test")], 
    [("vqa_v2","train"), ("vqa_cp", "test")], 
    [("vqa_v2","train"), ("vqa_ce", "test")], 
    [("vqa_v2","train"), ("vqa_rephrasings", "test")],

    # [("vqa_v2","train"), ("vqa_vs", "id_val")], 
    # [("vqa_v2","train"), ("vqa_v2","test")],

    # [("vqa_v2","train"), ("vqa_vs", "KO")],
    # [("vqa_v2","train"), ("vqa_vs", "KOP")],
    # [("vqa_v2","train"), ("vqa_vs", "KW")],
    # [("vqa_v2","train"), ("vqa_vs", "KW_KO")],
    # [("vqa_v2","train"), ("vqa_vs", "KWP")],
    # [("vqa_v2","train"), ("vqa_vs", "QT")],
    # [("vqa_v2","train"), ("vqa_vs", "QT_KO")],
    # [("vqa_v2","train"), ("vqa_vs", "QT_KW")],
    # [("vqa_v2","train"), ("vqa_vs", "QT_KW_KO")]
    # [("vqa_v2", "train"), ("vqa_cp", "test")],
    # [("vqa_v2", "train"), ("vqa_lol", "test")]
]


#store result vector for each tensor size (n_samples, 1) in order index
if __name__ == "__main__" : 
        
    for measure_instance in splits : 
        print(torch.cuda.memory_allocated())
        print("Memory summary", torch.cuda.memory_summary(device=None, abbreviated=False))
        
        test_ds_name, test_split = measure_instance[1]
        test_ds_split = f"{test_ds_name}_{test_split}"

        test_file = ds_split_2_file[test_ds_split]
        print(f"Attention Matrix {test_ds_split}") 

        with open(test_file, 'r') as f : 
            test_data = json.load(f)
  
        dataset = MeasureAttnDataset(test_data, test_ds_name)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn, num_workers=2)

        #VQAV2 -> store results of torch
        img_ratio = []
        txt_ratio = []
        image_path, question, answer = [], [], []
        instance_id = []

        for batch in tqdm(dataloader) : 
            img_ratio_batch, txt_ratio_batch = model.attn_scores(batch)
            img_ratio += img_ratio_batch
            txt_ratio += txt_ratio_batch
            image_path += batch["image_path"]
            question += batch["text_input_raw"]
            answer += batch["multiple_choice_answer"]
            instance_id += batch["instance_id"]
        
        # save the two lists to a csv file in /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xttn_results
        csv_file = f"/coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xttn_results/instance_{test_ds_split}.csv"
        
        df = pd.DataFrame({
            "instance_id" : instance_id,
            "image_path" : image_path,
            "question" : question,
            "answer" : answer,
            "img_ratio" : img_ratio,
            "txt_ratio" : txt_ratio
        })
        df.to_csv(csv_file, index=True)
    
        torch.cuda.empty_cache()
        gc.collect()
                