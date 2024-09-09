import torch
import pandas as pd
import numpy as np
import os

class RGN():
    def __init__(self):
        pass

    def get_param_groups(self, model):
        param_groups = {}
        for k, param in model.named_parameters():
            if param.requires_grad:
                param_groups[k] = {
                        "group_name": k,
                        "params": [],
                        }
                param_groups[k]["params"] = [param]
        self.keys = param_groups.keys()
        self.rgn = {k:0 for k in self.keys}
        return param_groups

    def calculate_rate(self, model):
        max_val = 0
        for k, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
                gradient_norm = torch.norm(param.grad).cpu()
            else:
                gradient_norm = torch.zeros(1).squeeze()
            param_norm = torch.norm(param.detach()).cpu()
            self.rgn[k] = gradient_norm/param_norm
            if self.rgn[k] > max_val: max_val = self.rgn[k]
            
        # Normalize 
        for k in self.keys:
            self.rgn[k] = self.rgn[k]/(max_val+ 1e-8)

    def adjust_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * self.rgn[param_group["group_name"]]
            
    def save(self,output_dir):
        save_data = {k:[v.cpu().numpy()] for k,v in self.rgn.items()}
        df = pd.DataFrame(save_data)
        df.to_csv(os.path.join(output_dir,'lr_ratio.csv'), index=False, mode="a",header=False)