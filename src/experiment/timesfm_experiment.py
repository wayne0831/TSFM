##########################################################################################################
# import libraries
###########################################################################################################

import ast
import gc
import os
import time
import ast
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

def run_lora_experiment(data_name, tsfm_method, ft_method, tr_data, te_context, cl, hl, patch_size, 
                        rank, alpha, dropout, target_modules, batch_size, lr, epochs):
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

    print(f"Loading LoRA...")
    tsfm.model = apply_lora_to_tsfm(
        model = tsfm.model,
        target_modules = target_modules,
        rank = rank,
        alpha = alpha,
        dropout = dropout
        )
    tsfm.model.to(DEVICE)

    # set trainining set and dataloader
    train_dataset = TimeSeriesDataset(tr_data, cl=cl, hl=hl, patch_size=patch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train the LoRA-adapted model
    print(f"\n Training for {epochs} epochs...")
    train_start_time = time.time()
    
    tsfm_lora, history = train(tsfm, train_loader, hl, patch_size, lr, epochs)
    
    total_train_time = time.time() - train_start_time
    avg_epoch_time = total_train_time / epochs

    print("\n" + "="*50)
    print(f"✅ Training Completed!")
    print(f"Total Training Time: {total_train_time:.2f}s ({total_train_time/60:.2f} mins)")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    print("="*50)

    # # build tsfm model
    tsfm_lora.compile(tsfm_config)

    print("Predicting with TimesFM + LoRA...")
    start_inf_time = time.time()
    with torch.no_grad():
        lora_preds, lora_actuals = forecast(model_obj=tsfm_lora, data=te_context, cl=cl, hl=hl, patch_size=patch_size)
    # end with
    inf_time = time.time() - start_inf_time
    print(f"TimesFM + LoRA Inference Time: {inf_time:.2f}s")

    # save predictions as .npy format
    pred_save_path = RES_PATH['predictions'][ft_method]
    pred_file_name = f"{tsfm_method}_{data_name}_cl{cl}_hl{hl}_{ft_method}_fr{ft_ratio}_r{rank}_a{alpha}_d{dropout}_tgt{target_modules}_bs{batch_size}_lr{lr}_e{epochs}preds.npy"
    pred_npy_save_path  = pred_save_path + pred_file_name

    np.save(pred_npy_save_path, lora_preds)
    print(f"✅ Predictions saved to: {pred_npy_save_path}")    

    # calcualte performance metrics
    mae, mse, wape, smape = calculate_metrics(lora_actuals, lora_preds)

    # save results as .csv format
    res_save_path = RES_PATH['performance'][ft_method]
    res_file_name = f"{tsfm_method}_{ft_method}_performance.csv"
    res_csv_save_path = res_save_path + res_file_name
    
    res_data = {
        'data': data_name, 'method': tsfm_method, 'cl': cl, 'hl': hl, 
        'ft_method': ft_method, 'ft_ratio': ft_ratio, 'rank': rank, 'alpha': alpha, 'dropout': dropout, 
        'target_modules': target_modules, 'batch_size': batch_size, 'lr': lr, 'epochs': epochs, 
        'mae': mae, 'mse': mse, 'wape': wape, 'smape': smape, 'tr_time': total_train_time, 'inf_time': inf_time
    }
    
    res_df = pd.DataFrame([res_data])
    file_exists = os.path.isfile(res_csv_save_path)
    res_df.to_csv(res_csv_save_path, mode='a', header=not file_exists, index=False)
    
    print(f"✅Performance metrics saved to: {res_csv_save_path}")


if __name__ == "__main__":
    # set common configurations
    data_name   = DATA

    # set TSFM-specific configurations
    tsfm_method = TSFM_METHOD
    patch_size  = PARAMS[tsfm_method]['patch_size']
    cl = PARAMS[TSFM_METHOD]['cl']
    hl = PARAMS[TSFM_METHOD]['hl']

    # set fine-tuning specific configurations
    ft_method   = FT_METHOD
    ft_ratio    = PARAMS['FT_RATIO']
    rank        = PARAMS[ft_method]['rank']
    alpha       = PARAMS[ft_method]['alpha']
    dropout     = PARAMS[ft_method]['dropout']
    target_modules = PARAMS[ft_method]['target_modules']
    batch_size  = PARAMS[ft_method]['batch_size']
    lr          = PARAMS[ft_method]['lr']

    # common configuration list for all experiments
    tsfm_method_list = [x.strip() for x in tsfm_method.split(',')]
    
    # TSFM configuration list
    data_name_list   = [x.strip() for x in data_name.split(',')]
    path_size_list   = [int(x.strip()) for x in patch_size.split(',')]
    cl_list = [int(x.strip()) for x in cl.split(',')]
    hl_list = [int(x.strip()) for x in hl.split(',')]

    # Fine-tuning configuration
    ft_method_list = [x.strip() for x in ft_method.split(',')]
    ft_ratio_list  = [float(x.strip()) for x in ft_ratio.split(',')]
    rank_list      = [int(x.strip()) for x in rank.split(',')]
    alpha_list     = [int(x.strip()) for x in alpha.split(',')]
    dropout_list   = [float(x.strip()) for x in dropout.split(',')]
    target_modules_list = ast.literal_eval(target_modules)
    batch_size_list = [int(x.strip()) for x in batch_size.split(',')]
    lr_list         = [float(x.strip()) for x in lr.split(',')]    
    epochs_list     = [int(x.strip()) for x in PARAMS[ft_method]['epochs'].split(',')]

    tsfm_comb = list(product(data_name_list, tsfm_method_list, path_size_list, ft_ratio_list, cl_list, hl_list))
    tsfm_total_comb = len(tsfm_comb)

    # # run TimesFM exepriemnt
    # for idx, (dn_item, tm_item, ps_item, fr_item, cl_item, hl_item) in enumerate(tsfm_comb, 1):
    #     print("\n" + "="*60)        
    #     print(f"Experiment [{idx} / {tsfm_total_comb}]") 
    #     print(f'data_name: {dn_item}, tsfm_method: {tm_item}, patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}')
    
    #     # load raw data
    #     df_path = DATA_PATH[dn_item]
    #     tgt_col = DATASET[dn_item]['target_col']
    #     df_raw  = pd.read_csv(df_path)

    #     # set target data and split fine tuning / test set
    #     target = df_raw[tgt_col].values.astype(np.float32)
    #     ft_len = int(len(target) * fr_item)

    #     tr_data = target[:ft_len] 
    #     te_data = target[ft_len:] 
    #     te_context = target[ft_len - cl_item:]

    #     run_timesfm_experiment(dn_item, tm_item, te_context, cl_item, hl_item, ps_item)
    # # end for

    tsfm_lora_comb = list(product(data_name_list, tsfm_method_list, ft_method_list, path_size_list, ft_ratio_list, cl_list, hl_list, 
                                  rank_list, alpha_list, dropout_list, target_modules_list, batch_size_list, lr_list, epochs_list))
    tsfm_lora_total_comb = len(tsfm_lora_comb)

    for idx, (dn_item, tm_item, ft_item, ps_item, fr_item, cl_item, hl_item, rank_item, alpha_item, dropout_item, target_modules_item, batch_size_item, lr_item, epochs_item) in enumerate(tsfm_lora_comb, 1):
        print("\n" + "="*60)        
        print(f"Experiment [{idx} / {tsfm_lora_total_comb}]") 
        print(f'data_name: {dn_item}, tsfm_method: {tm_item}, ft_method: {ft_item}, patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}')
        print(f'rank: {rank_item}, alpha: {alpha_item}, dropout: {dropout_item}, target_modules: {target_modules_item}, batch_size: {batch_size_item}, lr: {lr_item}, epochs: {epochs_item}')

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

        run_lora_experiment(data_name=dn_item, tsfm_method=tm_item, ft_method=ft_item, tr_data=tr_data, te_context=te_context, cl=cl_item, hl=hl_item, patch_size=ps_item, 
                            rank=rank_item, alpha=alpha_item, dropout=dropout_item, target_modules=target_modules_item, batch_size=batch_size_item, lr=lr_item, epochs=epochs_item)

        # [추가] 메모리 강제 해제
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2) # GPU가 정리될 시간을 잠시 줍니다.
    # end for
    