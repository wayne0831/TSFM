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

def sliding_window_forecast(model_obj, data, context_len, horizon_len):
    predictions = []
    actuals = []
    
    # i=0부터 시작하여 데이터의 극초반부(과거 데이터가 부족한 시점)도 예측 가능하게 합니다.
    for i in range(0, len(data) - horizon_len + 1, horizon_len):
        # 1. 현재 시점(i) 기준으로 과거 데이터(context) 추출
        start_idx = max(0, i - context_len)
        context_raw = data[start_idx : i]
        actual = data[i : i + horizon_len]
        
        # 2. 패딩 및 마스크 초기화 (context_len 크기)
        context = np.zeros(context_len, dtype=np.float32)
        mask = np.zeros(context_len, dtype=np.float32)
        
        # 3. 데이터가 있는 부분만 채우고 마스크 표시 (뒷부분부터 채우는 Pre-padding)
        if len(context_raw) > 0:
            context[-len(context_raw):] = context_raw
            mask[-len(context_raw):] = 1.0
            
        # 4. TimesFM 2.5의 3차원 입력 요구사항 대응 [1, seq_len, 1]
        # 모델의 forecast 메서드가 내부적으로 처리하지 못할 경우를 대비해 차원을 맞춥니다.
        context_input = context.reshape(1, context_len, 1)
        mask_input = mask.reshape(1, context_len, 1)

        # 5. 모델 예측 호출
        # TimesFM API 규격에 따라 inputs와 masks를 리스트나 텐서로 전달합니다.
        try:
            # 일반적인 forecast API 호출
            forecast_output, _ = model_obj.forecast(
                horizon=horizon_len, 
                inputs=context_input,
                masks=mask_input
            )
            predictions.extend(forecast_output[0])
        except TypeError:
            # 만약 forecast 메서드가 masks 인자를 직접 받지 않는 버전이라면
            # 내부 torch 모델을 직접 호출하여 처리합니다.
            device = next(model_obj.model.parameters()).device
            inputs_ts = torch.tensor(context_input).to(device)
            masks_ts = torch.tensor(mask_input).to(device)
            
            with torch.no_grad():
                # outputs[0] shape: [batch, horizon, 1]
                outputs = model_obj.model(inputs=inputs_ts, masks=masks_ts)
                pred_values = outputs[0][0, :horizon_len, 0].cpu().numpy()
                predictions.extend(pred_values)
        
        actuals.extend(actual)
        
    return np.array(predictions), np.array(actuals)

def calculate_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    
    return mae, mse, rmse, mape