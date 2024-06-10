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
from lavis.models import load_preprocess
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
from lavis.common.registry import registry
from data.infoseek import BLIP2Dataset, evaluate_model


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu", freeze_vit=False):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)

    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        # load model
        cfg.model.freeze_vit = freeze_vit
        model_cfg = cfg.model
        model = model_cls.from_config(model_cfg)

        if is_eval:
            model.eval()
        
        # load preprocess
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val_seen", help="val, test, or human")
    parser.add_argument("--name", type=str, default="blip2_t5", help="blip2_t5 | blip2_t5_instruct | blip2_opt | blip2_vicuna_instruct")
    parser.add_argument("--model_type", type=str, default="pretrain_flant5xxl", help="pretrain_flant5xxl ｜ flant5xxl ｜ pretrain_opt2.7b")
    parser.add_argument("--output_dir", type=str, default="predictions", help="output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="accumulation size")
    parser.add_argument("--use_lora", type=int, help="use lora")
    parser.add_argument("--target_modules", type=str, default=None, nargs='*', help="target modules")
    parser.add_argument("--ratio", type=str, default="10%", help="ratio")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--early_stop", type=int, default=20, help="early stop")
    parser.add_argument("--val_print_freq", type=int, default=1000, help="val print freq")
    parser.add_argument("--epoch", type=int, default=10, help="epoch")
    parser.add_argument("--opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
    parser.add_argument("--best_model_task", type=str, default=None, help="best model task")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora alpha")
    parser.add_argument("--lora_rank", type=int, default=16, help="lora rank")
    parser.add_argument("--adamh_wd", type=float, default=0.3, help="adamh weight decay")

    args = parser.parse_args()

    set_seed(args.seed)
    set_logger(args.output_dir + "/train.log")
    logging.info("Initialize Processor...")
    
    if args.ratio == "100%":
        split2data = {
            "val_seen": "infoseek/infoseek_val_seen.jsonl",
            "val_unseen": "infoseek/infoseek_val_unseen.jsonl",
            "test_seen": "infoseek/infoseek_test_seen.jsonl",
            "test_unseen": "infoseek/infoseek_test_unseen.jsonl",
            "train": "infoseek/infoseek_train.jsonl"
        }
    else:
        split2data = {
            "val_seen": f"infoseek/infoseek_val_seen_{args.ratio}.jsonl",
            "val_unseen": f"infoseek/infoseek_val_unseen_{args.ratio}.jsonl",
            "test_seen": "infoseek/infoseek_test_seen.jsonl",
            "test_unseen": "infoseek/infoseek_test_unseen.jsonl",
            "train": f"infoseek/infoseek_train_{args.ratio}.jsonl"
        }

    id2path = dict()

    # load image paths
    with open(ID2IMAGE, "r") as f:
        for line in f:
            line = json.loads(line)
            image_id = line["image_id"]
            path = line["image_path"]
            id2path[image_id] = path

    model, vis_processors, _ = load_model_and_preprocess(name=args.name,
                                                         model_type=args.model_type, 
                                                         is_eval=False, 
                                                         device="cuda")
    
    logging.info("target modules: {}".format(args.target_modules))
    logging.info(f"if use lora: {args.use_lora}")  
    logging.info(f"lora alpha: {args.lora_alpha}")
    logging.info(f"lora rank: {args.lora_rank}")
    logging.info(f"optimizer: {args.opt}")

    if (args.use_lora == 0) and (args.target_modules == ['v', 'q', 'qkv']):
        logging.info("train all parameters")
        args.batch_size = int(args.batch_size / 8)
        args.accumulation_steps = int(args.accumulation_steps * 8)
        logging.info(f"batch size: {args.batch_size}")
        logging.info(f"accumulation steps: {args.accumulation_steps}")
    
    if (args.use_lora == 0) and (args.target_modules == ['v', 'q']):
        logging.info("train V Q parameters")
        args.batch_size = int(args.batch_size / 8)
        args.accumulation_steps = int(args.accumulation_steps * 8)
        logging.info(f"batch size: {args.batch_size}")
        logging.info(f"accumulation steps: {args.accumulation_steps}")
    
    if args.use_lora == 1:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=args.target_modules,  # ['v', 'q', 'qkv'],  # qformer, qkv
        )
        
        logging.info(config)
        model = get_peft_model(model, config)
        

    # raw_image = Image.open("aircraft.png").convert("RGB")
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to("cuda")
    # output = model.generate({"image": image, "prompt": "Question: what is the date this aircraft took the first flight? Answer:"})
    # print(output)

    blip_dataset = BLIP2Dataset(
        split="train",
        processor=vis_processors,
        split2data=split2data,
        id2path=id2path,
        PROMPT="Question: {} Short answer:"
    )
    logging.info("Initialize Dataloader...")
    # Padding dataloader
    train_dataloader = DataLoader(
        blip_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
    )

    # # freeze everything except qformer
    logging.info("Freeze Model...")
    for name, param in model.named_parameters():
        if "Qformer" in name:
            param.requires_grad = True
        else:
            if args.use_lora == 0:
                if_freeze = True
                for target_module in args.target_modules:
                    if f".{target_module}." in name:
                        param.requires_grad = True
                        if_freeze = False
                        logging.info(name)
                        break
                if if_freeze:
                    param.requires_grad = False
    
    # use lora to train the visual and text encoder
    if args.use_lora == 1:
        logging.info(model.print_trainable_parameters())

    # optmizer adamw for all parameters require grad
    # optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if args.opt == "adam":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    else:
        optimizer_params = {
            "lr": args.lr,
            "weight_decay": args.wd,
            "momentum": 0.9,
            "nesterov": True,
        }   
        optimizer = torch.optim.SGD(trainable_params, **optimizer_params)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    accum_iter = args.accumulation_steps

    writer = SummaryWriter(args.output_dir)
    optimization_step = 0
    best_val_score = 0
    best_model_name = None
    early_stop = args.early_stop
    early_stop_flag = False
    
    for epoch in range(args.epoch):
        start_time = time.time()
        train_loss = AverageMeter("train_loss", ":.4e")

        logging.info(f"=============== Epoch: {epoch} ===============")
        for idx, batch in enumerate(tqdm(train_dataloader)):
            batch["image"] = batch["image"].squeeze(1).to(device)
            output = model(samples=batch)
            loss = output["loss"]
            train_loss.update(loss.item(), batch["image"].size(0))
            # Gradient accumulation
            loss = loss / accum_iter
            loss.backward()
            # print(loss.item())
            if (idx + 1) % accum_iter == 0 or idx == len(train_dataloader) - 1:
                optimization_step += 1
                optimizer.step()
                optimizer.zero_grad()

                if (optimization_step + 1) % args.val_print_freq == 0 or idx == len(train_dataloader) - 1:
                    writer.add_scalar("loss/train_loss", train_loss.avg, optimization_step)
                    
                    logging.info(f"Step: {optimization_step} | Train Loss: {train_loss.avg}")
                    
                    logging.info("Evaluation...")
                    model.eval()
                    val_result = evaluate_model(split=args.split, model=model, batch_size=args.batch_size, step=optimization_step, prompt="Question: {} Short answer:",
                                                args=args, epoch=epoch, split2data=split2data, id2path=id2path, vis_processors=vis_processors, device=device)      
                    # logging.info("Step:", idx)
                    logging.info(f"Validation result: {val_result}")
                    if args.split == "val_seen":
                        cur_val_score = val_result["seen_score"]["score"]
                    else:
                        cur_val_score = val_result["final_score"]
                    
                    writer.add_scalar("score/val_score", cur_val_score, optimization_step)
                    if args.split == "val_unseen":
                        writer.add_scalar("score/val_unseen_question_score", val_result["unseen_question_score"]["score"], optimization_step)
                        writer.add_scalar("score/val_unseen_entity_score", val_result["unseen_entity_score"]["score"], optimization_step)
                    
                    if cur_val_score > best_val_score:
                        best_val_score = cur_val_score
                        best_model_name = f"{args.output_dir}/{args.name}_{args.model_type}_{optimization_step}_val={cur_val_score}_epoch={epoch}.pt"
                        early_stop = args.early_stop
                        torch.save(model.state_dict(), best_model_name)
                        logging.info("-------- Save Best Model! --------")
                    else:
                        early_stop -= 1
                        logging.info("Early Stop Left: {}".format(early_stop))
                    if early_stop == 0:
                        logging.info("-------- Early Stop! --------")
                        early_stop_flag = True
                        break
                    model.train()

        if early_stop_flag:
            break
        
        # logging epoch finished in xx hours xx minutes xx seconds
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        logging.info(f"Epoch {epoch} finished in {elapsed_time}")
    
    # load best model according to best model name

    best_models = {
        "zeroshot": None,
        "Q": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora0_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_9999_val=66.89_epoch=2.pt", 
        "VQ": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetqkv_20240426_013428/blip2_t5_pretrain_flant5xxl_19999_val=72.45_epoch=4.pt",
        "QL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q_20240426_013428/blip2_t5_pretrain_flant5xxl_38488_val=79.46_epoch=7.pt",
        "VQL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_24055_val=79.28_epoch=4.pt",

        "Q_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora0_targetv q qkv_20240426_044130/blip2_t5_pretrain_flant5xxl_48110_val=77.79_epoch=9.pt",
        "VQ_ftp": "experiments/experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora1_targetqkv_20240513_180700/blip2_t5_pretrain_flant5xxl_91409_val=80.15_epoch=18.pt",
        "QL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q_20240426_044133/blip2_t5_pretrain_flant5xxl_48110_val=79.42_epoch=9.pt",
        "VQL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q qkv_20240426_044132/blip2_t5_pretrain_flant5xxl_48110_val=78.7_epoch=9.pt",

        "Q_h": "experiments/wd_0.3_new/experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora0_target_20240515_052826/blip2_t5_pretrain_flant5xxl_48110_val=77.94_epoch=9.pt",
        "VQ_h": "experiments/wd_0.1_new/experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora1_targetqkv_20240514_122104/blip2_t5_pretrain_flant5xxl_43299_val=77.7_epoch=8.pt",
        "QL_h": "experiments/experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora1_targetv q_20240513_180528/blip2_t5_pretrain_flant5xxl_86598_val=81.38_epoch=17.pt",
        "VQL_h": "experiments/wd_0.3_new/experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora1_targetv q qkv_20240515_052827/blip2_t5_pretrain_flant5xxl_43299_val=80.2_epoch=8.pt",
    }
    
    if args.epoch == 0:  # use best model
        if args.opt == "adam":
            tmp = ""
        elif args.opt == "adamp":
            tmp = "_ftp"
        elif args.opt == "adamh":
            tmp = "_h"
        
        if args.use_lora == 1:
            if args.target_modules == ["v", "q", "qkv"]:
                args.best_model_task = "VQL" + tmp
            elif args.target_modules == ["v", "q"]:
                args.best_model_task = "QL" + tmp
            elif args.target_modules == ["qkv"]:
                args.best_model_task = "VQ" + tmp
        else:
            args.best_model_task = "Q" + tmp

    if args.best_model_task is not None:
        logging.info("best model task: {}".format(args.best_model_task))
        model.load_state_dict(torch.load(best_models[args.best_model_task], map_location='cpu'))
    if args.epoch > 0:  # not zero-shot
        logging.info("load best model name and not zero-shot")
        model.load_state_dict(torch.load(best_model_name, map_location='cpu'))
    model.eval()

    if args.epoch == 0 and args.best_model_task is None:
        logging.info("Zero-shot evaluation ...")
    
    logging.info("Validation seen ...")
    val_seen_result = evaluate_model(split="val_seen", model=model, batch_size=args.batch_size, step=0, prompt="Question: {} Short answer:",
                                args=args, epoch=0, split2data=split2data, id2path=id2path, vis_processors=vis_processors, device=device)
    logging.info(f"Validation seen result: {val_seen_result}")

    logging.info("Validation unseen ...")
    val_unseen_result = evaluate_model(split="val_unseen", model=model, batch_size=args.batch_size, step=0, prompt="Question: {} Short answer:",
                                args=args, epoch=0, split2data=split2data, id2path=id2path, vis_processors=vis_processors, device=device)
    logging.info(f"Validation unseen result: {val_unseen_result}")
    
    # logging.info("Testing ...")
    # test_seen_result = evaluate_model(split="test_seen", model=model, batch_size=args.batch_size, step=0, prompt="Question: {} Short answer:",
    #                             args=args, epoch=0, split2data=split2data, id2path=id2path, vis_processors=vis_processors, device=device)
    # logging.info(f"Testing result (seen): {test_seen_result}")
    
    # test_unseen_result = evaluate_model(split="test_unseen", model=model, batch_size=args.batch_size, step=0, prompt="Question: {} Short answer:",  
    #                             args=args, epoch=0, split2data=split2data, id2path=id2path, vis_processors=vis_processors, device=device)
    # logging.info(f"Testing result (unseen): {test_unseen_result}")