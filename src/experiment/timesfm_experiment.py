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

def run_timesfm_experiment(data_name, tsfm_method, tr_data, te_data, cl, hl, patch_size):
    # data scaling
    scaler = TimeSeriesScaler()
    tr_data_scl = scaler.fit_transform(tr_data)
    te_data_scl = scaler.transform(te_data)
    print(f"Data Statistics: Mean={scaler.mean:.4f}, Std={scaler.std:.4f}")
    print(f"Tr scaled: Mean={tr_data_scl.mean():.4f}, Std={tr_data_scl.std():.4f} | Te scaled: Mean={te_data_scl.mean():.4f}, Std={te_data_scl.std():.4f}")

    # set tsfm model
    model_ver = PARAMS[tsfm_method]['version']
    
    print(f"Loading TimesFM: {model_ver}...")
    tsfm = TimesFM_2p5_200M_torch.from_pretrained(model_ver)
    tsfm_config = ForecastConfig(
        max_context=cl, 
        max_horizon=hl, 
        use_continuous_quantile_head=True, 
        normalize_inputs=False # 위에 data scaling을 수행하고 여러 기법들 간 공정한 성능 비교를 위해 False로 세팅
    )

    # build tsfm model
    tsfm.compile(tsfm_config)
    tsfm.model.to(DEVICE)

    # predict test set with tsfm
    print("Predicting with TimesFM...")
    start_inf_time = time.time()
    base_preds_scl, base_actuals_scl = forecast(
        model_obj=tsfm, 
        data=te_data_scl, 
        cl=cl, 
        hl=hl, 
        patch_size=patch_size
    )
    # reverse scaling to get predictions and actuals in original scale
    base_preds   = scaler.inverse_transform(base_preds_scl)
    base_actuals = scaler.inverse_transform(base_actuals_scl)

    print(f"# of predictions: {len(base_preds)}, # of actuals: {len(base_actuals)}")
    print(f"Predictions: {base_preds[:10]}, # actuals: {base_actuals[:10]}")
    print(f'Scaled predictions: {base_preds_scl[:10]}, Scaled actuals: {base_actuals_scl[:10]}')    

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
    mae_scl, mse_scl, wape_scl, smape_scl = calculate_metrics(base_actuals_scl, base_preds_scl)

    # save results as .csv format
    res_save_path = RES_PATH['performance'][tsfm_method]
    res_file_name = f"{tsfm_method}_performance.csv"
    res_csv_save_path = res_save_path + res_file_name
    
    res_data = {
        'data': data_name, 'method': tsfm_method, 'cl': cl, 'hl': hl, 
        'mae': mae, 'mse': mse, 'wape': wape, 'smape': smape, 'inf_time':inf_time,
        'mae_scl': mae_scl, 'mse_scl': mse_scl, 'wape_scl': wape_scl, 'smape_scl': smape_scl
    }
    
    print(f"Performance Metrics: MAE={round(mae, 4)}, MSE={round(mse, 4)}, WAPE={round(wape, 2)}%, sMAPE={round(smape, 2)}%")
    print(f"Performance Metrics (Scaled): MAE={round(mae_scl, 4)}, MSE={round(mse_scl, 4)}, WAPE={round(wape_scl, 2)}%, sMAPE={round(smape_scl, 2)}%")

    res_df = pd.DataFrame([res_data])
    file_exists = os.path.isfile(res_csv_save_path)
    res_df.to_csv(res_csv_save_path, mode='a', header=not file_exists, index=False)
    
    print(f"✅Performance metrics saved to: {res_csv_save_path}")

def run_lora_experiment(data_name, tsfm_method, ft_method, tr_data, te_data, cl, hl, patch_size, 
                        rank, alpha, dropout, target_modules, batch_size, lr, epochs):
    
    # data scaling
    scaler = TimeSeriesScaler()
    tr_data_scl = scaler.fit_transform(tr_data)
    te_data_scl = scaler.transform(te_data)
    print(f"Data Statistics: Mean={scaler.mean:.4f}, Std={scaler.std:.4f}")
    print(f"Tr scaled: Mean={tr_data_scl.mean():.4f}, Std={tr_data_scl.std():.4f} | Te scaled: Mean={te_data_scl.mean():.4f}, Std={te_data_scl.std():.4f}")

    # set tsfm model
    model_ver = PARAMS[tsfm_method]['version']
    print(f"Loading TimesFM: {model_ver}...")
    tsfm = TimesFM_2p5_200M_torch.from_pretrained(model_ver)

    tsfm_config = ForecastConfig(
        max_context=cl, 
        max_horizon=hl, 
        use_continuous_quantile_head=True, 
        normalize_inputs=False # 위에 data scaling을 수행하고 여러 기법들 간 공정한 성능 비교를 위해 False로 세팅
    )

    # apply LoRA to TimesFM
    print(f"Loading LoRA...")
    tsfm.model, tr_params_ratio = apply_lora_to_tsfm(
        model = tsfm.model,
        target_modules = target_modules,
        rank = rank,
        alpha = alpha,
        dropout = dropout
    )
    tsfm.model.to(DEVICE)

    # set dataloader
    train_dataset = TimeSeriesDataset(tr_data_scl, cl=cl, hl=hl, patch_size=patch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train model
    print(f"\n Training for {epochs} epochs...")
    train_start_time = time.time()
    
    tsfm_lora, history = train(
        model=tsfm, 
        train_loader=train_loader, 
        max_horizon=hl, 
        patch_size=patch_size, 
        lr=lr, 
        epochs=epochs
    )
    
    total_train_time = time.time() - train_start_time
    avg_epoch_time = total_train_time / epochs

    print("\n" + "="*50)
    print(f"✅ Training Completed!")
    print(f"Total Training Time: {total_train_time:.2f}s")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    print("="*50)

    # 6. 추론 및 역정규화 (Inverse Transform)
    tsfm_lora.compile(tsfm_config)
    print("Predicting with TimesFM + LoRA...")
    start_inf_time = time.time()
    
    with torch.no_grad():
        # 정규화된 컨텍스트로 예측 수행
        lora_preds_scl, lora_actuals_scl = forecast(
            model_obj=tsfm_lora, 
            data=te_data_scl, 
            cl=cl, hl=hl, 
            patch_size=patch_size
        )

    # reverse scaling to get predictions and actuals in original scale
    lora_preds   = scaler.inverse_transform(lora_preds_scl)
    lora_actuals = scaler.inverse_transform(lora_actuals_scl)

    print(f"# of predictions: {len(lora_preds)}, # of actuals: {len(lora_actuals)}")
    print(f"Predictions: {lora_preds[:5]}, # actuals: {lora_actuals[:5]}")
    print(f'Scaled predictions: {lora_preds_scl[:5]}, Scaled actuals: {lora_actuals_scl[:5]}')    

    inf_time = time.time() - start_inf_time

    print(f"TimesFM + LoRA Inference Time: {inf_time:.2f}s")

    # 7. 성능 지표 계산 및 저장
    mae, mse, wape, smape = calculate_metrics(lora_actuals, lora_preds)
    mae_scl, mse_scl, wape_scl, smape_scl = calculate_metrics(lora_actuals_scl, lora_preds_scl)

    pred_save_path = RES_PATH['predictions'][ft_method]
    pred_file_name = f"{tsfm_method}_{data_name}_cl{cl}_hl{hl}_{ft_method}_r{rank}_a{alpha}_d{dropout}_tgt{target_modules}_lr{lr}_e{epochs}_bs{batch_size}_preds.npy"
    pred_npy_save_path = os.path.join(pred_save_path, pred_file_name)

    np.save(pred_npy_save_path, lora_preds)
    print(f"✅ Real-scale Predictions saved to: {pred_npy_save_path}")    

    res_save_path = RES_PATH['performance'][ft_method]
    res_file_name = f"{tsfm_method}_{ft_method}_performance.csv"
    res_csv_save_path = os.path.join(res_save_path, res_file_name)
    
    res_data = {
        'data': data_name, 'method': tsfm_method, 'cl': cl, 'hl': hl, 
        'ft_method': ft_method, 'rank': rank, 'alpha': alpha, 'dropout': dropout, 
        'target_modules': str(target_modules), 'lr': lr, 'epochs': epochs, 'batch_size': batch_size, 
        'mae': mae, 'mse': mse, 'wape': wape, 'smape': smape, 
        'mae_scl': mae_scl, 'mse_scl': mse_scl, 'wape_scl': wape_scl, 'smape_scl': smape_scl,
        'tr_time': total_train_time, 'inf_time': inf_time, 'tr_params_ratio': tr_params_ratio
    }
    
    print(f"Performance Metrics: MAE={round(mae, 4)}, MSE={round(mse, 4)}, WAPE={round(wape, 2)}%, sMAPE={round(smape, 2)}%")
    print(f"Performance Metrics (Scaled): MAE={round(mae_scl, 4)}, MSE={round(mse_scl, 4)}, WAPE={round(wape_scl, 2)}%, sMAPE={round(smape_scl, 2)}%")

    res_df = pd.DataFrame([res_data])
    file_exists = os.path.isfile(res_csv_save_path)
    res_df.to_csv(res_csv_save_path, mode='a', header=not file_exists, index=False)
    
    # 8. 메모리 해제 및 구조 원복
    print('🔄 Reverting model structure and clearing cache...')
    tsfm.model = remove_lora_from_tsfm(tsfm.model)
    del tsfm
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
        print(f"현재 디바이스: {torch.cuda.get_device_name(0)}")
    
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

    # set TimesFM combinations for all experiments
    tsfm_comb = list(product(data_name_list, tsfm_method_list, path_size_list, ft_ratio_list, cl_list, hl_list))
    num_tsfm_comb = len(tsfm_comb)

    # set LoRA combinations for all experiments
    lora_comb = list(product(data_name_list, tsfm_method_list, ft_method_list, path_size_list, ft_ratio_list, cl_list, hl_list, 
                            rank_list, alpha_list, dropout_list, target_modules_list, batch_size_list, lr_list, epochs_list))
    num_lora_comb = len(lora_comb)

    # run TimesFM exepriemnt
    if PIPELINE['TimesFM']:
        for idx, (dn_item, tm_item, ps_item, fr_item, cl_item, hl_item) in enumerate(tsfm_comb, 1):
            print("\n" + "="*60)        
            print(f"Experiment [{idx} / {num_tsfm_comb}]") 
            print(f'data_name: {dn_item}, tsfm_method: {tm_item}, patch_size: {ps_item}, ft_ratio: {fr_item}, cl: {cl_item}, hl: {hl_item}')
        
            # load raw data
            df_path = DATA_PATH[dn_item]
            tgt_col = DATASET[dn_item]['target_col']
            df_raw  = pd.read_csv(df_path)

            # set target data and split fine tuning / test set
            # target = df_raw[tgt_col].values.astype(np.float32)
            target = df_raw[tgt_col].fillna(0).values.astype(np.float32)
            ft_len = int(len(target) * fr_item)

            tr_data = target[:ft_len] 
            te_data = target[ft_len - cl_item:] # context 길이만큼 겹치도록 설정하여 모델이 test set의 초기 부분을 보고 예측을 시작할 수 있도록 함

            run_timesfm_experiment(data_name=dn_item, tsfm_method=tm_item, 
                                   tr_data=tr_data, te_data=te_data, cl=cl_item, hl=hl_item, patch_size=ps_item)
        # end for
    # end if
        
    if PIPELINE['LoRA']:
        for idx, (dn_item, tm_item, ft_item, ps_item, fr_item, cl_item, hl_item, rank_item, alpha_item, dropout_item, target_modules_item, batch_size_item, lr_item, epochs_item) in enumerate(lora_comb, 1):
            print("\n" + "="*60)        
            print(f"Experiment [{idx} / {num_lora_comb}]") 
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
            te_data = target[ft_len - cl_item:] # context 길이만큼 겹치도록 설정하여 모델이 test set의 초기 부분을 보고 예측을 시작할 수 있도록 함

            run_lora_experiment(data_name=dn_item, tsfm_method=tm_item, ft_method=ft_item, 
                                tr_data=tr_data, te_data=te_data, cl=cl_item, hl=hl_item, patch_size=ps_item, 
                                rank=rank_item, alpha=alpha_item, dropout=dropout_item, target_modules=target_modules_item, 
                                batch_size=batch_size_item, lr=lr_item, epochs=epochs_item)
        # end for
    # end if