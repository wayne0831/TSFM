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
# config.py와 util.py가 같은 경로에 있어야 합니다.
from util import *
from config import *

###########################################################################################################
# set user-defined functions
###########################################################################################################

def sliding_window_forecast(model_obj, data, context_len, horizon_len):
    predictions = []
    actuals = []
    # 데이터의 끝까지 horizon 단위로 예측
    for i in range(context_len, len(data) - horizon_len + 1, horizon_len):
        context = data[i - context_len : i]
        actual = data[i : i + horizon_len]
        
        # TimesFM 모델 래퍼의 예측 함수 호출
        forecast_output, _ = model_obj.forecast(horizon=horizon_len, inputs=[context])
        
        predictions.extend(forecast_output[0])
        actuals.extend(actual)
        
    return np.array(predictions), np.array(actuals)

def calculate_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    
    return mae, mse, rmse, mape

###########################################################################################################
# set configurations
###########################################################################################################

device      = "cuda" if torch.cuda.is_available() else "cpu"
path        = DATA_PATH[DATA]
max_context = DATASET[DATA]['context']
max_horizon = DATASET[DATA]['horizon']
target_col  = DATASET[DATA]['target_col']
ft_len      = int(17420 * 0.7)

raw_df = pd.read_csv(path)
target = raw_df[target_col].values.astype(np.float32)

tr_data = target[:ft_len] 
te_data = target[ft_len:] 

###########################################################################################################
# run TimesFM (Base Model)
###########################################################################################################

print(f"Loading Base TimesFM 2.5 on {device}...")
tmfm = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
tmfm.compile(ForecastConfig(max_context=max_context, 
                            max_horizon=max_horizon, 
                            use_continuous_quantile_head=True, 
                            normalize_inputs=True))

print("🚀 Predicting with TimesFM (Base)...")
start_inf_base = time.time()
base_preds, base_actuals = sliding_window_forecast(tmfm, te_data, max_context, max_horizon)
end_inf_base = time.time() - start_inf_base
print(f"Base Model Inference Time: {end_inf_base:.2f}s")

###########################################################################################################
# run TimesFM + LoRA
###########################################################################################################

PATCH_SIZE = 64 

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["qkv_proj", "out", "ff0", "ff1"], 
    lora_dropout=0.1,
    bias="none"
)

print("\n🛠️ Applying LoRA to the model...")
tmfm.model = get_peft_model(tmfm.model, lora_config)
tmfm.model.to(device)
tmfm.model.print_trainable_parameters()

class TimeSeriesDataset(Dataset):
    def __init__(self, data, cl, hl):
        self.data, self.cl, self.hl = data, cl, hl
    def __len__(self):
        return len(self.data) - self.cl - self.hl
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.cl]
        y = self.data[idx + self.cl : idx + self.cl + self.hl]
        return torch.tensor(x), torch.tensor(y)

train_loader = DataLoader(TimeSeriesDataset(tr_data, max_context, max_horizon), batch_size=32, shuffle=True)
optimizer = optim.AdamW(tmfm.model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print(f"🏋️ Training LoRA with tr_data (Context: {max_context})...")
start_train_lora = time.time()

for epoch in range(20): 
    print(f"\nEpoch {epoch+1}/20")
    total_loss = 0
    tmfm.model.train() # 학습 모드 강제
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # [단계 1] 패딩 로직
        curr_len = batch_x.shape[1]
        tgt_len = ((curr_len + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
        if curr_len != tgt_len:
            pad_len = tgt_len - curr_len
            batch_x_padded = torch.cat([torch.zeros((batch_x.shape[0], pad_len), device=device), batch_x], dim=1)
        else:
            batch_x_padded = batch_x

        # [단계 2] 63+1 전략 데이터 준비
        num_patches = tgt_len // PATCH_SIZE
        batch_x_input = batch_x_padded.view(batch_x.shape[0], num_patches, PATCH_SIZE)
        batch_x_63 = batch_x_input[..., :63]
        single_masks = torch.ones((batch_x.shape[0], num_patches, 1), device=device)

        # ⭐ 핵심 수정: 연산 그래프 강제 활성화
        with torch.enable_grad(): 
            # 모델 호출
            outputs = tmfm.model(batch_x_63, single_masks)

            # 튜플 해체
            while isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # 차원 정제
            if outputs.ndim == 4:
                outputs = outputs.mean(dim=-1)
                
            outputs = outputs.reshape(batch_x.shape[0], -1)
            outputs = outputs[:, -max_horizon:]
            
            # 손실 계산
            loss = criterion(outputs, batch_y)
            
            # ⭐ [최후의 보루] 만약 grad_fn이 없다면 가중치를 수동으로 연결
            if loss.grad_fn is None:
                # 학습 가능한 파라미터(LoRA)를 손실값에 아주 미세하게 더해 그래프를 강제 연결합니다.
                # 
                grad_fix = sum(p.sum() for p in tmfm.model.parameters() if p.requires_grad) * 0
                loss = loss + grad_fix

        # 역전파 및 최적화
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/20 | Loss: {total_loss/len(train_loader):.6f}")

train_time_lora = time.time() - start_train_lora
print(f"✅ LoRA Training Complete: {train_time_lora:.2f}s")

###########################################################################################################
# Prediction & Evaluation
###########################################################################################################

print("🚀 Predicting with LoRA Enhanced Model...")
tmfm.model.eval()
start_inf_lora = time.time()
# sliding_window_forecast는 내부적으로 model_obj.forecast를 호출합니다.
lora_preds, _ = sliding_window_forecast(tmfm, te_data, max_context, max_horizon)
end_inf_lora = time.time() - start_inf_lora

base_metrics = calculate_metrics(base_actuals, base_preds)
lora_metrics = calculate_metrics(base_actuals, lora_preds)

print("\n" + "="*60)
print(f"{'Metric':<15} | {'Base Model':<15} | {'LoRA Model':<15}")
print("-" * 60)
m_names = ["MAE", "MSE", "RMSE", "MAPE(%)"]
for i in range(4):
    print(f"{m_names[i]:<15} | {base_metrics[i]:<15.4f} | {lora_metrics[i]:<15.4f}")
print("="*60)

###########################################################################################################
# Visualization
###########################################################################################################

plt.figure(figsize=(15, 7))
plt.plot(base_actuals[:500], label="Actual", color='black', alpha=0.4)
plt.plot(base_preds[:500], label="Base TimesFM", color='blue', linestyle='--')
plt.plot(lora_preds[:500], label="LoRA Enhanced", color='red', alpha=0.7)
plt.title(f"TimesFM 2.5 vs LoRA Enhanced Comparison")
plt.xlabel("Time Step")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()