###########################################################################################################
# import libraries
###########################################################################################################

import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, PeftModel
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import *

###########################################################################################################
# set user-defined functions
###########################################################################################################

# define custom Dataset for time-series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, cl, hl, patch_size):
        self.data = data
        self.cl, self.hl, self.p_size = int(cl), int(hl), int(patch_size)

    def __len__(self):
        return len(self.data) - self.cl - self.hl

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.cl] 
        y = self.data[idx + self.cl : idx + self.cl + self.hl]
        
        target_cl = ((self.cl + self.p_size - 1) // self.p_size) * self.p_size
        x_padded = np.zeros(target_cl, dtype=np.float32)
        x_padded[-len(x):] = x 
        
        mask = np.zeros(target_cl, dtype=np.float32)
        mask[:target_cl - len(x)] = 1

        return torch.tensor(x_padded), torch.tensor(y), torch.tensor(mask)

class TimeSeriesScaler:
    def __init__(self):
        self.mean = self.std = None
    def fit_transform(self, data):
        self.mean, self.std = np.mean(data), np.std(data) + 1e-8
        return (data - self.mean) / self.std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def calculate_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred)**2)
    wape = (np.sum(np.abs(actual - pred)) / (np.sum(np.abs(actual)) + 1e-8)) * 100
    smape = np.mean(np.abs(actual - pred) / ((np.abs(actual) + np.abs(pred)) / 2 + 1e-8)) * 100
    return round(mae, 4), round(mse, 4), round(wape, 2), round(smape, 2)

def forecast(model_obj, data, cl, hl, patch_size):
    predictions, actuals = [], []
    p_size = int(patch_size)
    target_cl = ((cl + p_size - 1) // p_size) * p_size
    
    # LoRA 레이어 유무 확인
    is_lora = any("LoRALayer" in str(type(m)) for m in model_obj.model.modules())

    i = cl 
    while i < len(data):
        rem_len = min(hl, len(data) - i)
        ctx_raw, actual = data[i - cl : i], data[i : i + rem_len]
        
        if is_lora:
            if i == cl: print(f"🔍 [Mode] Manual inference (LoRA-tuned)")
            ctx_padded = np.zeros(target_cl, dtype=np.float32)
            ctx_padded[-len(ctx_raw):] = ctx_raw
            
            num_patches = target_cl // p_size
            inputs_ts = torch.tensor(ctx_padded).view(1, num_patches, p_size).to(DEVICE)
            
            mask_np = np.zeros(target_cl, dtype=np.float32)
            mask_np[:target_cl - len(ctx_raw)] = 1
            masks_ts = torch.tensor(mask_np).view(1, num_patches, p_size).to(DEVICE)
            
            with torch.no_grad():
                outputs = model_obj.model(inputs_ts, masks_ts)
                while isinstance(outputs, (tuple, list)): 
                    outputs = outputs[0]
                
                if outputs.dim() == 4:
                    all_preds = outputs[0, :, :, 0].reshape(-1)
                else:  # 3차원일 경우
                    all_preds = outputs[0].reshape(-1)
                    
                # [수정] 슬라이싱 버그 해결
                pred_values = all_preds[-hl:][:rem_len].cpu().numpy()
        else:
            if i == cl: print(f"🔍 [Mode] Standard TimesFM")
            f_out, _ = model_obj.forecast(inputs=[ctx_raw], horizon=rem_len)
            pred_values = f_out[0]

        predictions.extend(pred_values)
        actuals.extend(actual)
        i += hl

    return np.array(predictions), np.array(actuals)