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
from util import *
from LoRA import *
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
# run TimesFM + LoRA (Final Fixed Version - Mask None)
###########################################################################################################

if PIPELINE.get('TimesFM_LoRA', True):
    print("\n" + "="*50)
    print("🚀 Starting TimesFM + LoRA Fine-tuning (Clean Version)...")
    print("="*50)

    # 1. 모델 로드 및 LoRA 주입
    lora_params = LORA_PARAMS[DATA]
    tmfm_lora_model = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
    tmfm_lora_model.model = apply_lora_to_tsfm(
        model = tmfm_lora_model.model,
        target_modules = lora_params['target_modules'],
        rank = lora_params['r'],
        alpha = lora_params['alpha'],
        dropout = lora_params['dropout']
        )
    tmfm_lora_model.model.to(device)

    tsfm_params = TSFM_PARAMS[DATA]
    max_context = tsfm_params['max_context']
    max_horizon = tsfm_params['max_horizon']
    
    # 2. Dataset 준비
    train_dataset = TimeSeriesDataset(tr_data, cl=max_context, hl=max_horizon)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. 학습 설정
    trainable_params = [p for p in tmfm_lora_model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.MSELoss()

    # 4. Training Loop
    epochs = 10
    tmfm_lora_model.model.train()
    
    print(f"⏱️ Starting training for {epochs} epochs...")
    train_start_time = time.time()  # 전체 학습 시작 시간 측정

    for epoch in range(epochs):
        epoch_start_time = time.time()  # 개별 에포크 시작 시간
        epoch_loss = 0
        
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            optimizer.zero_grad()
            
            num_patches = batch_x.shape[1] // TSFM_PATCH_SIZE
            x_reshaped = batch_x.view(-1, num_patches, TSFM_PATCH_SIZE)
            
            x_63 = x_reshaped[:, :, :63]
            mask_1 = torch.zeros(x_reshaped.shape[0], num_patches, 1).to(device)
            
            outputs = tmfm_lora_model.model(x_63, mask_1)
            
            while isinstance(outputs, (tuple, list)): 
                outputs = outputs[0]
            logits = outputs

            if logits.dim() == 4:
                pred_all = logits[:, :, :, 0].reshape(batch_x.shape[0], -1)
            else:
                pred_all = logits.reshape(batch_x.shape[0], -1)
                
            pred_last = pred_all[:, -max_horizon:]
            
            if pred_last.shape[1] != batch_y.shape[1]:
                loss = criterion(pred_last, batch_y[:, :pred_last.shape[1]])
            else:
                loss = criterion(pred_last, batch_y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_duration = time.time() - epoch_start_time # 에포크 소요 시간
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss/len(train_loader):.6f} | Time: {epoch_duration:.2f}s")

    total_train_time = time.time() - train_start_time # 전체 소요 시간
    avg_epoch_time = total_train_time / epochs
    
    print("\n" + "="*50)
    print(f"✅ Training Completed!")
    print(f"Total Training Time: {total_train_time:.2f}s ({total_train_time/60:.2f} mins)")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")
    print("="*50)

    # 5. 예측 수행 (Inference)
    print("🚀 Predicting with TimesFM + LoRA...")
    tmfm_lora_model.model.eval()
    
    # Inference 시에도 내부적으로 동일한 forward 로직을 타도록 구성된 유틸 함수 호출
    # (주의: sliding_window_forecast 내부에서도 model 호출 시 mask에 None을 넣어야 함)
    tmfm_config = ForecastConfig(
        max_context=max_context, 
        max_horizon=max_horizon, 
        use_continuous_quantile_head=True, 
        normalize_inputs=True
    )
    tmfm_lora_model.compile(tmfm_config)

    start_inf_lora = time.time()
    with torch.no_grad():
        # [주의] 아래 함수는 내부적으로 try/except를 통해 LoRA 모델 전용 None 마스크 추론을 수행합니다.
        test_input_data = target[ft_len - max_context:]
        lora_preds, lora_actuals = sliding_window_forecast(
            model_obj=tmfm_lora_model, 
            data=test_input_data, 
            cl=max_context, 
            hl=max_horizon
        )
    end_inf_lora = time.time() - start_inf_lora
    print(f"LoRA Model Inference Time: {end_inf_lora:.2f}s")

    # 6. 결과 시각화 및 저장
    # (기존 시각화 로직 유지)
    plot_path = RES_PATH['plot'].get('timesfm_lora_plot', 'timesfm_lora_res.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ LoRA Result saved to: {plot_path}")

    # 6. 결과 시각화 및 저장
    start_idx = ft_len
    pred_idx = np.arange(start_idx, start_idx + len(lora_preds))

    plt.figure(figsize=(15, 7))
    plt.plot(target, label="Actual (True)", color='black', alpha=0.4, linewidth=1)
    plt.plot(pred_idx, lora_preds, label="TimesFM + LoRA Prediction", color='blue', linestyle='--', linewidth=1.2)
    
    if 'base_preds' in locals():
        plt.plot(pred_idx, base_preds, label="Base TimesFM", color='red', alpha=0.5)

    plt.axvline(x=ft_len, color='blue', linestyle=':', label='Test Set Index')
    plt.title(f"Actual vs TimesFM+LoRA prediction on {DATA}")
    plt.xlabel("Time Step")
    plt.ylabel(tgt_col)
    plt.legend()
    plt.grid(True, alpha=0.2)

    plot_path = RES_PATH['plot'].get('timesfm_lora_plot', 'timesfm_lora_res.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    npy_path = RES_PATH['array'].get('timesfm_lora_preds', 'lora_preds.npy')
    np.save(npy_path, lora_preds)
    print(f"✅ LoRA Result saved to: {plot_path} and {npy_path}")

    # 7. 성능 비교 리포트 출력
    print("\n" + "="*75)
    print("📈 Calculating Final Performance Metrics (Base vs LoRA)")
    print("="*75)
    
    # 지표 계산
    b_mae, b_mse, b_rmse, b_wape, b_smape = calculate_metrics(base_actuals, base_preds)
    l_mae, l_mse, l_rmse, l_wape, l_smape = calculate_metrics(lora_actuals, lora_preds)

    performance_report = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'WAPE (%)', 'sMAPE (%)'],
        'Base TimesFM': [b_mae, b_mse, b_rmse, b_wape, b_smape],
        'TimesFM + LoRA': [l_mae, l_mse, l_rmse, l_wape, l_smape]
    })

    # 개선율 계산
    performance_report['Improvement (%)'] = (
        (performance_report['Base TimesFM'] - performance_report['TimesFM + LoRA']) / 
        (performance_report['Base TimesFM'] + 1e-8) * 100
    ).round(2)

    print(performance_report.to_string(index=False))
    print("="*75)

    # # CSV 결과 저장 (보고서 및 논문용)
    # report_save_path = f"./results/performance_report_{DATA}.csv"
    # performance_report.to_csv(report_save_path, index=False)
    # print(f"✅ Performance report saved to: {report_save_path}")

    # # 간단한 분석 코멘트 출력
    # if lora_mse < base_mse:
    #     improvement = ((base_mse - lora_mse) / base_mse) * 100
    #     print(f"💡 LoRA Fine-tuning reduced MSE by {improvement:.2f}% compared to Base model.")
    # else:
    #     print("💡 Fine-tuning did not improve MSE in this run. Consider adjusting hyperparams.")