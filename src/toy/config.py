###########################################################################################################
# import libraries
###########################################################################################################

import sys
import pandas as pd
import numpy as np

###########################################################################################################
# set version configurations
###########################################################################################################

DATE      = '260220' # date
MODEL_VER = 'google/timesfm-2.5-200m-pytorch' # FM model verision
DATA      = 'Etth1'  # dataset name

###########################################################################################################
# set path configurations
###########################################################################################################

# data path
DATA_PATH = {
    'Etth1': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
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
}

###########################################################################################################
# set hyperparameter configurations
###########################################################################################################

TIMESFM_HYPERPARAMS = {
    'base': {
        'max_context': 96, 
        'max_horizon': 192,
    },
    'Etth1': {
        'max_context': 96,
        'max_horizon': 192,
    },
}

LORA_HYPERPARAMS = {
    'base': {
        'patch_size': 64,
        'epoch': 5,
        'batch_size': 64,
        'lr': 1e-3,
        'r': 4,
        'lora_alpha': 16,
        'target_modules': ".*stacked_xf.*qkv_proj",
        'lora_dropout': 0.1,
        'bias': "none",
    },
    'Etth1': {
        'patch_size': 64,
        'epoch': 5,
        'batch_size': 64,
        'lr': 1e-3,
        'r': 4,
        'lora_alpha': 16,
        'target_modules': ["attn.qkv_proj", "attn.out"],
        'lora_dropout': 0.1,
        'bias': "none",
    },
}

HYPERPARAMS = {
    'TimesFM': {
        'base': {
            'max_context': 96,
            'max_horizon': 192,
        },
        'Etth1': {
            'max_context': 96,
            'max_horizon': 192,
        },
    },
    'LoRA': {
        'base': {
            'r': 4,
            'lora_alpha': 16,
            'target_modules': ["attn.qkv_proj", "attn.out", "ff0", "ff1"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
        'Etth1': {
            'r': 4,
            'lora_alpha': 16,
            'target_modules': ["attn.qkv_proj", "attn.out", "ff0", "ff1"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
    },
    'RL_LoRA': {
        'state':  [],
        'action': [],
        'reward': [],
    }
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

if __name__ == "__main__":
    print(f"📊 Loading {DATA} data...")
    df = pd.read_csv(DATA_PATH[DATA])

    print(df.head(), len(df))
    #data_values = df[target_col].values.astype(np.float32)