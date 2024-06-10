import os
import time
import pandas as pd
import itertools
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import copy
import torch
from PIL import Image
import json
from infoseek_eval import evaluate as evaluate_infoseek
from infoseek_eval import evaluate_seen
import argparse
from infoseek_data.data_path import INFOSEEK_SPLIT2DATA, ID2IMAGE, IMAGES, OVEN_SPLIT2DATA
from peft import LoraConfig, get_peft_model
from utils import set_logger, AverageMeter
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from omegaconf import OmegaConf

def create_eval_data(split, split2data, id2path):
    # Read the input JSONL file
    with open(split2data[split], 'r') as f:
        batch_data = [json.loads(line) for line in f]

    clean_batch_data = []
    not_exit = []
    for idx, item in enumerate(batch_data):
        if idx % 10000 == 0:
            logging.info(f"Processing {idx}/{len(batch_data)}")
        path = id2path[item["image_id"]]
        # check path exists
        if not os.path.exists(path):
            not_exit.append(item["image_id"])
        else:
            clean_batch_data.append(item)
    return clean_batch_data

def load_and_process_image(item, vis_processors, device, id2path):
    # Load and preprocess the image
    raw_image = Image.open(id2path[item["image_id"]]).convert("RGB").resize((224, 224))    
    processed_image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return processed_image, item["question"], item["data_id"]

def process_images_in_batches(model, batch_data, batch_size, prompt, vis_processors, device, id2path):
    # Create a pool of workers
    # Monitor the progress of the pool
    
    output = []
    logging.info("Generate predictions...")
    # Process images in batches
    for idx, i in enumerate(range(0, len(batch_data), batch_size)):
        if (idx + 1) % 100 == 0:
            logging.info(f"Processing batch {idx}/{len(batch_data)/batch_size}")
        # Subset results for the current batch
        batch_subset = batch_data[i:i+batch_size]

        # Separate the images, questions, and ids
        batch_images, batch_questions, batch_ids = [], [], []

        # Load and preprocess the images
        for item in batch_subset:
            tmp_img, tmp_q, tmp_id = load_and_process_image(item, vis_processors, device, id2path)
            batch_images.append(tmp_img)
            batch_questions.append(tmp_q)
            batch_ids.append(tmp_id)

        # Concatenate the batch images
        image_batch = torch.cat(batch_images, dim=0)
        
        # add prompt to questions
        batch_questions = [prompt.format(q) for q in batch_questions]
        # Generate predictions for the batch
        
        answers = model.generate({"image": image_batch, "prompt": batch_questions},
                                 length_penalty=-1)  # default: num_beams=5
        # print(batch_questions)
        # print(answers)
        
        for idx, ans in zip(batch_ids, answers):
            output.append({"data_id": idx, "prediction": ans})
    return output

def evaluate_model(split, model, batch_size, step, prompt, args, epoch, split2data, id2path, vis_processors, device):
    # Create evaluate data
    batch_data = create_eval_data(split, split2data, id2path)
    # Process the data in batches
    output = process_images_in_batches(model, batch_data, batch_size, prompt, vis_processors, device, id2path)

    # Save the predictions
    # development_{args.batch_size}_all_lora
    pred_path = f"{args.output_dir}/{args.name}_{args.model_type}_{split}_{step}_epoch={epoch}.jsonl"
    
    # ref_path = f"infoseek_data/infoseek_{split}.jsonl"
    ref_path = split2data[split]
    # ref_qtype_path = f"infoseek_data/infoseek_{split}_qtype.jsonl"
    
    with open(pred_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    if split == "val_seen" or split == "test_seen":
        result = evaluate_seen(pred_path, ref_path)
    else:
        result = evaluate_infoseek(pred_path, ref_path, ref_path)
    return result


class BLIP2Dataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, split2data, id2path, PROMPT="Question: {} Short answer:"):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.image_path = []
        self.question = []
        self.answer = []
        with open(split2data[split], "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                image_id = line["image_id"]
                path = id2path[image_id]
                self.image_path.append(path)
                self.question.append(line["question"])
                self.answer.append(line["answer"][0])

        self.vis_processor = processor
        self.prompt = PROMPT
 
    def __getitem__(self, idx):
        raw_image = Image.open(self.image_path[idx]).convert("RGB").resize((224, 224))
        question = self.prompt.format(self.question[idx])
        answer = self.answer[idx]
        processed_image = self.vis_processor["train"](raw_image).unsqueeze(0)
        inputs = {"image": processed_image, "text_input": question, "text_output": answer}
        return inputs
 
    def __len__(self):
        return len(self.question)