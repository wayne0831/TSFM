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
import inspect

###########################################################################################################
# set configurations
###########################################################################################################
# set device and load data
device = "cuda" if torch.cuda.is_available() else "cpu"

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

print(f"Loading Base TimesFM 2.5 on {device}...")

max_context = TIMESFM_HYPERPARAMS[DATA]['max_context']
max_horizon = TIMESFM_HYPERPARAMS[DATA]['max_horizon']

tmfm_base   = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
tmfm_config = ForecastConfig(
    max_context=max_context, 
    max_horizon=max_horizon, 
    use_continuous_quantile_head=True, 
    normalize_inputs=True
)

tmfm_base.compile(tmfm_config)
tmfm_base.model.to(device)


# 모델 로드 후 (get_peft_model 호출 전)
for name, module in tmfm_base.model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)


"""
tokenizer.hidden_layer
tokenizer.output_layer
tokenizer.residual_layer
stacked_xf.0.attn.qkv_proj
stacked_xf.0.attn.out
stacked_xf.0.ff0
stacked_xf.0.ff1
stacked_xf.1.attn.qkv_proj
stacked_xf.1.attn.out
stacked_xf.1.ff0
stacked_xf.1.ff1
stacked_xf.2.attn.qkv_proj
stacked_xf.2.attn.out
stacked_xf.2.ff0
stacked_xf.2.ff1
stacked_xf.3.attn.qkv_proj
stacked_xf.3.attn.out
stacked_xf.3.ff0
stacked_xf.3.ff1
stacked_xf.4.attn.qkv_proj
stacked_xf.4.attn.out
stacked_xf.4.ff0
stacked_xf.4.ff1
stacked_xf.5.attn.qkv_proj
stacked_xf.5.attn.out
stacked_xf.5.ff0
stacked_xf.5.ff1
stacked_xf.6.attn.qkv_proj
stacked_xf.6.attn.out
stacked_xf.6.ff0
stacked_xf.6.ff1
stacked_xf.7.attn.qkv_proj
stacked_xf.7.attn.out
stacked_xf.7.ff0
stacked_xf.7.ff1
stacked_xf.8.attn.qkv_proj
stacked_xf.8.attn.out
stacked_xf.8.ff0
stacked_xf.8.ff1
stacked_xf.9.attn.qkv_proj
stacked_xf.9.attn.out
stacked_xf.9.ff0
stacked_xf.9.ff1
stacked_xf.10.attn.qkv_proj
stacked_xf.10.attn.out
stacked_xf.10.ff0
stacked_xf.10.ff1
stacked_xf.11.attn.qkv_proj
stacked_xf.11.attn.out
stacked_xf.11.ff0
stacked_xf.11.ff1
stacked_xf.12.attn.qkv_proj
stacked_xf.12.attn.out
stacked_xf.12.ff0
stacked_xf.12.ff1
stacked_xf.13.attn.qkv_proj
stacked_xf.13.attn.out
stacked_xf.13.ff0
stacked_xf.13.ff1
stacked_xf.14.attn.qkv_proj
stacked_xf.14.attn.out
stacked_xf.14.ff0
stacked_xf.14.ff1
stacked_xf.15.attn.qkv_proj
stacked_xf.15.attn.out
stacked_xf.15.ff0
stacked_xf.15.ff1
stacked_xf.16.attn.qkv_proj
stacked_xf.16.attn.out
stacked_xf.16.ff0
stacked_xf.16.ff1
stacked_xf.17.attn.qkv_proj
stacked_xf.17.attn.out
stacked_xf.17.ff0
stacked_xf.17.ff1
stacked_xf.18.attn.qkv_proj
stacked_xf.18.attn.out
stacked_xf.18.ff0
stacked_xf.18.ff1
stacked_xf.19.attn.qkv_proj
stacked_xf.19.attn.out
stacked_xf.19.ff0
stacked_xf.19.ff1
output_projection_point.hidden_layer
output_projection_point.output_layer
output_projection_point.residual_layer
output_projection_quantiles.hidden_layer
output_projection_quantiles.output_layer
output_projection_quantiles.residual_layer
"""