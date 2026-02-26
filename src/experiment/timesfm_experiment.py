##########################################################################################################
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
from util.util import *
from model.LoRA import *
from config import *

###########################################################################################################
# set configurations
###########################################################################################################
# set device and load data
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# load raw data
df_path = DATA_PATH[DATA]
tgt_col = DATASET[DATA]['target_col']
df_raw  = pd.read_csv(df_path)

# set target data and split train/test
target = df_raw[tgt_col].values.astype(np.float32)
ft_len = int(len(target) * 0.7)

tr_data = target[:ft_len] 
te_data = target[ft_len:] 

###########################################################################################################
# run TimesFM (Base Model)
###########################################################################################################

if PIPELINE['TimesFM']:
    print(f"Loading Base TimesFM 2.5 on {device}...")

    tsfm_params = TSFM_PARAMS[DATA]
    max_context = tsfm_params['max_context']
    max_horizon = tsfm_params['max_horizon']

    tmfm_base   = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
    tmfm_config = ForecastConfig(
        max_context=max_context, 
        max_horizon=max_horizon, 
        use_continuous_quantile_head=True, 
        normalize_inputs=True
    )

    tmfm_base.compile(tmfm_config)
    tmfm_base.model.to(device)

    print("🚀 Predicting with TimesFM (Base)...")
    # [수정] 테스트 데이터의 첫 번째 구간을 예측하기 위해 훈련 데이터의 마지막 컨텍스트를 포함
    test_input_data = target[ft_len - max_context:]
    start_inf_base = time.time()
    base_preds, base_actuals = sliding_window_forecast(
        model_obj=tmfm_base, 
        data=test_input_data, 
        cl=max_context, 
        hl=max_horizon
    )
    end_inf_base = time.time() - start_inf_base
    print(f"Base Model Inference Time: {end_inf_base:.2f}s")

    # visualize and save predictions
    start_idx = ft_len
    pred_idx  = np.arange(start_idx, start_idx + len(base_preds))

    # plot
    plt.figure(figsize=(15, 7))
    plt.plot(target, label="Actual (True)", color='black', alpha=0.4, linewidth=1)
    plt.plot(pred_idx, base_preds, label="TimesFM Prediction", color='red', linestyle='--', linewidth=1.2)
    plt.axvline(x=ft_len, color='blue', linestyle=':', label='Test Set Index')
    plt.axvline(x=start_idx, color='green', linestyle=':', label='Forecast Index')
    plt.title(f"Actual vs TimesFM prediction on {DATA} ")
    plt.xlabel("Time Step")
    plt.ylabel(tgt_col)
    plt.legend()
    plt.grid(True, alpha=0.2)

    # save plot
    plot_save_path = RES_PATH['plot']['timesfm_base_plot']
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_save_path}")
    #plt.show()

    npy_save_path = RES_PATH['array']['timesfm_base_preds']
    np.save(npy_save_path, base_preds)
    print(f"✅ Array saved to: {npy_save_path}")

    loaded_preds = np.load(npy_save_path)
    print(loaded_preds.shape, te_data.shape, loaded_preds)
# end if

###########################################################################################################
# set user-defined functions
###########################################################################################################

# 사전에 정해지는 것: device, data, ...

def run_timesfm_experiment(device, model, test_input_data, cl, hl):
    # set device and load data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load raw data
    df_path = DATA_PATH[DATA]
    tgt_col = DATASET[DATA]['target_col']
    df_raw  = pd.read_csv(df_path)

    # set target data and split train/test
    target = df_raw[tgt_col].values.astype(np.float32)
    ft_len = int(len(target) * 0.7)

    tr_data = target[:ft_len] 
    te_data = target[ft_len:] 
    pass