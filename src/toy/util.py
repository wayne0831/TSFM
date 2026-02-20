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
# config.py와 util.py가 같은 경로에 있어야 합니다.
from util import *
from config import *

###########################################################################################################
# set user-defined functions
###########################################################################################################

def sliding_window_forecast(model_obj, data, context_len, horizon_len):
    predictions = []
    actuals = []
    # 데이터의 끝까지 horizon 단위로 예측
    for i in range(context_len, len(data) - horizon_len + 1, horizon_len):
        context = data[i - context_len : i]
        actual  = data[i : i + horizon_len]
        
        # TimesFM 모델 래퍼의 예측 함수 호출
        forecast_output, _ = model_obj.forecast(horizon=horizon_len, inputs=[context])
        
        predictions.extend(forecast_output[0])
        actuals.extend(actual)
        
    return np.array(predictions), np.array(actuals)

def calculate_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    
    return mae, mse, rmse, mape