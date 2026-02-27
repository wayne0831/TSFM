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
from src.util.util import *
from src.model.LoRA import *
from config import *

###########################################################################################################
# set user-defined functions
###########################################################################################################

# 사전에 정해지는 것: device, data, ...

def run_timesfm_experiment(tsfm_method, te_context, cl, hl):
    # set tsfm model
    model_ver = PARAMS[tsfm_method]['version']
    
    print(f"Loading TimesFM: {model_ver}...")
    tsfm = TimesFM_2p5_200M_torch.from_pretrained(model_ver)
    tsfm_config = ForecastConfig(
        max_context=cl, 
        max_horizon=hl, 
        use_continuous_quantile_head=True, 
        normalize_inputs=True
    )

    # build tsfm model
    tsfm.compile(tsfm_config)
    tsfm.model.to(DEVICE)

    # predict test set with tsfm
    print("Predicting with TimesFM...")
    start_inf_time = time.time()
    base_preds, base_actuals = forecast(model_obj=tsfm, data=te_context, cl=cl, hl=hl, patch_size=patch_size)
    inf_time = time.time() - start_inf_time
    print(f"TimesFM Inference Time: {inf_time:.2f}s")
    
    # save predictions and actuals for later evaluation and plotting
    pred_save_path = RES_PATH['predictions'][tsfm_method]
    pred_file_name = f"{tsfm_method}_cl{cl}_hl{hl}_preds.npy"
    npy_save_path  = pred_save_path + pred_file_name

    np.save(npy_save_path, base_preds)
    print(f"✅ Predictions saved to: {npy_save_path}")    
    pass

if __name__ == "__main__":
    # for loop로 생성
    tsfm_method = 'TimesFM'
    #tsfm_ver    = 'TimesFM_2p5_200M_torch'
    patch_size  = PARAMS[tsfm_method]['patch_size'] # 64
    ft_method   = 'LoRA'
    
    data = 'Etth1'
    cl   = 96
    hl   = 192
    ft_ratio = 0.7

    # load raw data
    df_path = DATA_PATH[data]
    tgt_col = DATASET[data]['target_col']
    df_raw  = pd.read_csv(df_path)

    # set target data and split fine tuning / test set
    target = df_raw[tgt_col].values.astype(np.float32)
    ft_len = int(len(target) * ft_ratio)

    tr_data = target[:ft_len] 
    te_data = target[ft_len:] 
    te_context = target[ft_len - cl:]

    run_timesfm_experiment(tsfm_method=tsfm_method, te_context=te_context, cl=cl, hl=hl)