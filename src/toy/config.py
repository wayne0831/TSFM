###########################################################################################################
# import libraries
###########################################################################################################

import sys
import pandas as pd
import numpy as np

###########################################################################################################
# set version configurations
###########################################################################################################

DATE      = '260225' # date
MODEL_VER = 'google/timesfm-2.5-200m-pytorch' # FM model verision
DATA      = 'Etth1'  # dataset name

###########################################################################################################
# set path configurations
###########################################################################################################

# data path
DATA_PATH = {
    'Etth1': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
    'Etth2': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv',
    'Ettm1': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv',
    'Ettm2': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv',
}
CHK_PATH  = {
    'LoRA': f'./checkpoints/lora_{DATE}_{DATA}',
}
RES_PATH  = {
    'plot': {
        'timesfm_base_plot': f'./results/plot/timesfm_base_{DATE}_{DATA}.png',
        'timesfm_lora_plot': f'./results/plot/timesfm_lora_{DATE}_{DATA}.png',
    },
    'array': {
        'timesfm_base_preds': f'./results/array/timesfm_base_preds_{DATE}_{DATA}.npy',
        'timesfm_lora_preds': f'./results/array/timesfm_lora_preds_{DATE}_{DATA}.npy',
    }
}

###########################################################################################################
# set data configurations
###########################################################################################################

DATASET = {
    'Etth1': {
        'target_col': 'OT'
    },
    'Etth2': {
        'target_col': 'OT'
    },
    'Ettm1': {
        'target_col': 'OT'
    },
    'Ettm2': {
        'target_col': 'OT'
    },
}

###########################################################################################################
# set hyperparameter configurations
###########################################################################################################

TSFM_PATCH_SIZE = 64

TSFM_PARAMS = {
    'patch_size': 64,
    'base': {
        'max_context': 96, 
        'max_horizon': 192,
    },
    'Etth1': {
        'max_context': 96,
        'max_horizon': 192,
    },
    'Etth2': {
        'max_context': 96, 
        'max_horizon': 192,
    },
    'Ettm1': {
        'max_context': 96, 
        'max_horizon': 192,
    },
    'Ettm2': {
        'max_context': 96, 
        'max_horizon': 192,
    },
}

LORA_PARAMS = {
    'base': {
        'epoch': 5,
        'batch_size': 32,
        'lr': 1e-4,
        'r': 4,
        'alpha': 8,
        'target_modules': ["qkv_proj", "out", "ff0", "ff1"],
        'dropout': 0.1,
        'bias': "none",
    },
    'Etth1': {
        'epoch': 5,
        'batch_size': 32,
        'lr': 1e-4,
        'r': 4,
        'alpha': 8,
        'target_modules': ["qkv_proj", "out", "ff0", "ff1"],
        'dropout': 0.1,
        'bias': "none",
    },
    'Etth2': {
        'epoch': 5,
        'batch_size': 32,
        'lr': 1e-4,
        'r': 4,
        'alpha': 8,
        'target_modules': ["qkv_proj", "out", "ff0", "ff1"],
        'dropout': 0.1,
        'bias': "none",
    },
    'Ettm1': {
        'epoch': 5,
        'batch_size': 32,
        'lr': 1e-4,
        'r': 4,
        'alpha': 8,
        'target_modules': ["qkv_proj", "out", "ff0", "ff1"],
        'dropout': 0.1,
        'bias': "none",
    },
    'Ettm2': {
        'epoch': 5,
        'batch_size': 32,
        'lr': 1e-4,
        'r': 4,
        'alpha': 8,
        'target_modules': ["qkv_proj", "out", "ff0", "ff1"],
        'dropout': 0.1,
        'bias': "none",
    },
}



###########################################################################################################
# set pipeline
###########################################################################################################

PIPELINE = {
    'TimesFM':      True,
    'TimesFM_LoRA': True,
    'RL_LoRA':      False,
}

###########################################################################################################
# set model configurations
###########################################################################################################