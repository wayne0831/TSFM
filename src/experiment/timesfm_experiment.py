##########################################################################################################
# import libraries
###########################################################################################################

import os
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

def run_timesfm_experiment(data_name, tsfm_method, te_context, cl, hl, patch_size):
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
    print(f"# of predictions: {len(base_preds)}, # of actuals: {len(base_actuals)}")

    inf_time = time.time() - start_inf_time
    print(f"TimesFM Inference Time: {inf_time:.2f}s")
    
    # save predictions as .npy format
    pred_save_path = RES_PATH['predictions'][tsfm_method]
    pred_file_name = f"{tsfm_method}_{data_name}_cl{cl}_hl{hl}_preds.npy"
    pred_npy_save_path  = pred_save_path + pred_file_name

    np.save(pred_npy_save_path, base_preds)
    print(f"✅ Predictions saved to: {pred_npy_save_path}")    

    # calcualte performance metrics
    mae, mse, wape, smape = calculate_metrics(base_actuals, base_preds)

    # save results as .csv format
    res_save_path = RES_PATH['performance'][tsfm_method]
    res_file_name = f"{tsfm_method}_performance.csv"
    res_csv_save_path = res_save_path + res_file_name
    
    res_data = {
        'data': data_name, 'method': tsfm_method, 'cl': cl, 'hl': hl, 
        'mae': mae, 'mse': mse, 'wape': wape, 'smape': smape, 'inf_time': inf_time
    }
    
    res_df = pd.DataFrame([res_data])
    file_exists = os.path.isfile(res_csv_save_path)
    res_df.to_csv(res_csv_save_path, mode='a', header=not file_exists, index=False)
    
    print(f"✅Performance metrics saved to: {res_csv_save_path}")

if __name__ == "__main__":
    # for loop로 생성
    data_name   = DATA
    tsfm_method = TSFM_METHOD
    patch_size  = PARAMS[tsfm_method]['patch_size']
    ft_method   = FT_METHOD
    ft_ratio    = PARAMS['FT_RATIO']

    cl = PARAMS[TSFM_METHOD]['cl']
    hl = PARAMS[TSFM_METHOD]['hl']

    # TimesFM experiment
    data_name_list   = [x.strip() for x in data_name.split(',')]
    tsfm_method_list = [x.strip() for x in tsfm_method.split(',')]
    path_size_list   = [float(x.strip()) for x in patch_size.split(',')]
    ft_ratio_list    = [float(x.strip()) for x in ft_ratio.split(',')]
    cl_list = [int(x.strip()) for x in cl.split(',')]
    hl_list = [int(x.strip()) for x in hl.split(',')]

    combinations = list(product(data_name_list, tsfm_method_list, path_size_list, ft_ratio_list, cl_list, hl_list))
    total_comb = len(combinations)

    for idx, (dn_item, tm_item, ps_item, fr_item, cl_item, hl_item) in enumerate(combinations, 1):
        print("\n" + "="*60)        
        print(f"Experiment [{idx} / {total_comb}]") 
        print(f'data_name: {dn_item}, tsfm_method: {tm_item}, patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}')
    
        # load raw data
        df_path = DATA_PATH[dn_item]
        tgt_col = DATASET[dn_item]['target_col']
        df_raw  = pd.read_csv(df_path)

        # set target data and split fine tuning / test set
        target = df_raw[tgt_col].values.astype(np.float32)
        ft_len = int(len(target) * fr_item)

        tr_data = target[:ft_len] 
        te_data = target[ft_len:] 
        te_context = target[ft_len - cl_item:]

        run_timesfm_experiment(data_name=dn_item, tsfm_method=tm_item, te_context=te_context, cl=cl_item, hl=hl_item, patch_size=ps_item)
    # end for