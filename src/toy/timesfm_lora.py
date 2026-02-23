###########################################################################################################
# import libraries
###########################################################################################################

import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import * # DATA, DATA_PATH, TIMESFM_HYPERPARAMS 등 포함

###########################################################################################################
# set user-defined functions
###########################################################################################################

def sliding_window_forecast(model_obj, data, context_len, horizon_len):
    predictions = []
    actuals = []
    
    # TimesFM 고수준 API는 inputs 리스트를 받아 내부적으로 패딩/마스크를 자동 처리합니다.
    # 수동으로 마스크를 넣으면 패치 크기 불일치 에러가 날 수 있으므로 API에 맡깁니다.
    for i in range(0, len(data) - horizon_len + 1, horizon_len):
        start_idx = max(0, i - context_len)
        context_raw = data[start_idx : i]
        actual = data[i : i + horizon_len]
        
        # 데이터가 아예 없는 초기 시점 대응
        if len(context_raw) == 0:
            context_input = [np.zeros(1, dtype=np.float32)]
        else:
            context_input = [context_raw.astype(np.float32)]

        # forecast()는 masks 인자를 직접 받지 않으며 내부에서 자동 생성합니다.
        forecast_output, _ = model_obj.forecast(
            horizon=horizon_len, 
            inputs=context_input
        )
        
        predictions.extend(forecast_output[0])
        actuals.extend(actual)
        
    return np.array(predictions), np.array(actuals)

def calculate_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-9))) * 100
    return mae, mse, rmse, mape

###########################################################################################################
# set device and load data
###########################################################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

df_path = DATA_PATH[DATA]
tgt_col = DATASET[DATA]['target_col']
df_raw  = pd.read_csv(df_path)

target = df_raw[tgt_col].values.astype(np.float32)
ft_len = int(len(target) * 0.7)

tr_data = target[:ft_len] 
te_data = target[ft_len:] 

###########################################################################################################
# run TimesFM (Base Model)
###########################################################################################################

max_context = TIMESFM_HYPERPARAMS[DATA]['max_context']
max_horizon = TIMESFM_HYPERPARAMS[DATA]['max_horizon']

print(f"Loading Base TimesFM 2.5 on {device}...")
tmfm_base = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
tmfm_config = ForecastConfig(
    max_context=max_context, 
    max_horizon=max_horizon, 
    use_continuous_quantile_head=True, 
    normalize_inputs=True
)
tmfm_base.compile(tmfm_config)
tmfm_base.model.to(device)

if PIPELINE.get('TimesFM', True):
    print("🚀 Predicting with TimesFM (Base)...")
    start_inf_base = time.time()
    base_preds, base_actuals = sliding_window_forecast(
        model_obj=tmfm_base, 
        data=te_data, 
        context_len=max_context, 
        horizon_len=max_horizon
    )
    print(f"Base Model Inference Time: {time.time() - start_inf_base:.2f}s")

    # 결과 인덱스 계산
    start_idx = ft_len 
    pred_idx  = np.arange(start_idx, start_idx + len(base_preds))

###########################################################################################################
# run TimesFM + LoRA
###########################################################################################################

if PIPELINE.get('TimesFM_LoRA', True): 
    lora_param = LORA_HYPERPARAMS[DATA]
    epochs     = lora_param['epoch']

    # 1. LoRA 설정 및 적용
    lora_config = LoraConfig(
        r               = lora_param['r'],
        lora_alpha      = lora_param['lora_alpha'],
        target_modules  = lora_param['target_modules'], 
        lora_dropout    = lora_param['lora_dropout'],
        bias            = lora_param['bias']
    )

    print(f"🚀 Applying LoRA to TimesFM 2.5...")
    tmfm_lora_model = get_peft_model(tmfm_base.model, lora_config)
    tmfm_lora_model.print_trainable_parameters()

    # 2. 학습용 데이터셋 (고정 길이 패딩 사용)
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, context_len, horizon_len):
            self.data = data
            self.cl = context_len
            self.hl = horizon_len
        def __len__(self):
            return len(self.data) - self.hl + 1
        def __getitem__(self, idx):
            start_idx = max(0, idx - self.cl)
            x_raw = self.data[start_idx : idx]
            y_raw = self.data[idx : idx + self.hl]
            
            x = np.zeros(self.cl, dtype=np.float32)
            mask = np.zeros(self.cl, dtype=np.float32)
            if len(x_raw) > 0:
                x[-len(x_raw):] = x_raw
                mask[-len(x_raw):] = 1.0
                
            return (torch.tensor(x).unsqueeze(-1), 
                    torch.tensor(y_raw).unsqueeze(-1), 
                    torch.tensor(mask).unsqueeze(-1))

    # [Tip] max_context가 64의 배수일 때 모델 행렬 연산이 가장 안정적입니다.
    train_ds = TimeSeriesDataset(tr_data, max_context, max_horizon)
    train_loader = DataLoader(train_ds, batch_size=lora_param.get('batch_size', 32), shuffle=True)

    # 3. Fine-tuning 실행
    optimizer = optim.AdamW(tmfm_lora_model.parameters(), lr=lora_param.get('lr', 1e-4))
    criterion = nn.MSELoss()
    
    # TimesFM 2.5의 패치 사이즈 정의 (보통 64)
    PATCH_SIZE = 64 

    tmfm_lora_model.train()
    print(f"⌛ Starting Fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch, mask_batch in train_loader:
            x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
            
            # [수정 포인트] 패치 구조에 맞게 차원 재구성 (Reshape)
            # (Batch, Seq_len, 1) -> (Batch, Num_patches, Patch_size)
            # 예: (32, 192, 1) -> (32, 3, 64)
            num_patches = x_batch.shape[1] // PATCH_SIZE
            
            x_patched = x_batch.view(x_batch.shape[0], num_patches, PATCH_SIZE)
            mask_patched = mask_batch.view(mask_batch.shape[0], num_patches, PATCH_SIZE)
            
            optimizer.zero_grad()
            
            # 모델 호출 (패치된 데이터 전달)
            # 주의: 모델 구현에 따라 inputs/masks 인자 대신 직접 전달해야 할 수도 있음
            outputs = tmfm_lora_model(inputs=x_patched, masks=mask_patched) 
            
            # 손실 계산 (예측값과 실제값의 차원 일치 확인)
            loss = criterion(outputs[0][:, :max_horizon, :], y_batch) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.6f}")

    # 4. LoRA 모델로 예측 수행
    print("🚀 Predicting with TimesFM + LoRA...")
    tmfm_lora_model.eval()
    tmfm_base.model = tmfm_lora_model # 래퍼 내부 모델 교체
    
    start_inf_lora = time.time()
    lora_preds, lora_actuals = sliding_window_forecast(
        model_obj=tmfm_base, 
        data=te_data, 
        context_len=max_context, 
        horizon_len=max_horizon
    )
    print(f"LoRA Model Inference Time: {time.time() - start_inf_lora:.2f}s")

    # 5. 시각화 및 비교
    plt.figure(figsize=(15, 7))
    plt.plot(target, label="Actual", color='black', alpha=0.3)
    if PIPELINE.get('TimesFM', True):
        plt.plot(pred_idx, base_preds, label="Base TimesFM", color='red', linestyle='--', alpha=0.7)
    plt.plot(pred_idx, lora_preds, label="TimesFM + LoRA", color='green', linewidth=1.5)
    
    plt.axvline(x=ft_len, color='blue', linestyle=':', label='Test Set Start')
    plt.title(f"Comparison: Base vs LoRA on {DATA}")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(RES_PATH['plot']['timesfm_base_plot'].replace('base', 'lora_comp'), dpi=300)
    print(f"✅ Comparison plot saved.")

    # 6. 가중치 분석
    print("\n🔍 LoRA Weight Analysis (Norm):")
    for name, param in tmfm_lora_model.named_parameters():
        if 'lora_A' in name:
            print(f"  - {name:50} | Norm: {torch.norm(param).item():.6f}")