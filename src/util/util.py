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
    def __init__(self, data, cl, hl):
        self.data = data
        self.cl = cl # context length (e.g., 96)
        self.hl = hl # horizon length (e.g., 192)

    def __len__(self):
        return len(self.data) - self.cl - self.hl

    def __getitem__(self, idx, patch_size):
        # slice raw data for context and horizon 
        x = self.data[idx : idx + self.cl] 
        y = self.data[idx + self.cl : idx + self.cl + self.hl]
        
        # set padding length to the nearest multiple of TSFM_PATCH_SIZE (64)
        # cl: 96 -> target_cl: 128
        target_cl = ((self.cl + patch_size - 1) // patch_size) * patch_size
        
        # padding for input sequence (context) to match target_cl
        x_padded = np.zeros(target_cl, dtype=np.float32)
        x_padded[-len(x):] = x # set actual data at the end (right alignment)
        
        # generate mask for padded input (1 for padding, 0 for actual data)
        mask = np.zeros(target_cl, dtype=np.float32)
        mask[:target_cl - len(x)] = 1

        return torch.tensor(x_padded), torch.tensor(y), torch.tensor(mask)

# peform forecasting and collect predictions and actuals
def forecast(model_obj, data, cl, hl, patch_size):
    # Initialize lists to store predictions and ground truth values
    predictions, actuals = [], []

    # Calculate padding length to ensure compatibility with patch_size (64)
    # Example: cl=96 -> target_cl=128
    target_cl = ((cl + patch_size - 1) // patch_size) * patch_size
    
    i = cl 
    while i < len(data):
        # Determine the length of the current prediction window (handle end-of-series)
        remaining_len = min(hl, len(data) - i)
        
        # Extract the context (lookback) and the actual future values (ground truth)
        context_raw = data[i - cl : i]
        actual = data[i : i + remaining_len]
        
        try:
            # Standard inference path for the TimesFM model
            forecast_output, _ = model_obj.forecast(inputs=[context_raw], horizon=remaining_len)
            pred_values = forecast_output[0]
            
        except Exception:
            # Fallback path: Manual inference for LoRA-tuned models
            # Step 1: Pad the context to meet the model's patch-based input requirements
            context_padded = np.zeros(target_cl, dtype=np.float32)
            context_padded[-len(context_raw):] = context_raw # Right-align context
            
            # Step 2: Reshape input into [1, Num_Patches, Patch_Size]
            num_patches = target_cl // patch_size
            inputs_ts = torch.tensor(context_padded).view(1, num_patches, patch_size).to(DEVICE)
            
            # Step 3: Construct the padding mask (1 for padding, 0 for actual data)
            # Consistent with the logic used in TimeSeriesDataset
            mask_np = np.zeros(target_cl, dtype=np.float32)
            mask_np[:target_cl - len(context_raw)] = 1
            masks_ts = torch.tensor(mask_np).view(1, num_patches, patch_size).to(DEVICE)
            
            with torch.no_grad():
                # Step 4: Perform forward pass through the LoRA-adapted model
                outputs = model_obj.model(inputs_ts, masks_ts)
                
                # Step 5: Unpack potential tuple/list output structures from the model
                while isinstance(outputs, (tuple, list)): 
                    outputs = outputs[0]
                
                # Step 6: Extract point forecast (Channel 0) and reshape
                # Format: [Batch, Num_Patches, Patch_Size, Bins] -> [Total_Seq_Len]
                all_preds = outputs[0, :, :, 0].reshape(-1)
                
                # Step 7: Slice the relevant horizon window from the end of the predictions
                # Since TSFM predicts the next hl steps relative to the context
                pred_values = all_preds[-hl : -hl + remaining_len].cpu().numpy()

        # Collect results and advance the window by the horizon length
        predictions.extend(pred_values)
        actuals.extend(actual)
        i += hl

    return np.array(predictions), np.array(actuals)

# calculate performance metrics
def calculate_metrics(actual, pred):
    mae  = np.mean(np.abs(actual - pred))
    mse  = np.mean((actual - pred)**2)

    # WAPE (Weighted Absolute Percentage Error)
    # 전체 실제값의 크기 대비 전체 오차의 합을 측정하여 모델의 전반적인 정확도를 평가
    wape = (np.sum(np.abs(actual - pred)) / (np.sum(np.abs(actual)) + 1e-8)) * 100
    
    # 2sMAPE (Symmetric MAPE)
    # 분모에 실제값과 예측값의 평균을 사용하여 0~200% 사이의 값을 가지도록 정규화
    smape = np.mean(np.abs(actual - pred) / ((np.abs(actual) + np.abs(pred)) / 2 + 1e-8)) * 100
    
    return mae, mse, wape, smape