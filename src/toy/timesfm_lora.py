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
from config import *
from util import *

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
model = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
model.compile(ForecastConfig(max_context=max_context, 
                             max_horizon=max_horizon, 
                             use_continuous_quantile_head=True, 
                             normalize_inputs=True))

print("🚀 Predicting with TimesFM (Base)...")
start_inf_base = time.time()
base_preds, base_actuals = sliding_window_forecast(model, te_data, max_context, max_horizon)
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
model.model = get_peft_model(model.model, lora_config)
model.model.to(device)
model.model.print_trainable_parameters()

# [⭐ 에러 해결의 핵심] 토크나이저 입력 기대치 강제 조정
base_model = model.model.get_base_model()
if hasattr(base_model, 'tokenizer'):
    # 128 에러를 방지하기 위해 입력 피처를 64로 강제 고정
    base_model.tokenizer.hidden_layer.in_features = PATCH_SIZE
    # 현재 컨텍스트 길이에 맞춰 내부 설정 업데이트
    tgt_context = ((max_context + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    base_model.tokenizer.context_len = tgt_context

class TimeSeriesDataset(Dataset):
    def __init__(self, data, cl, hl):
        self.data, self.cl, self.hl = data, cl, hl
    def __len__(self):
        return len(self.data) - self.cl - self.hl
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx : idx + self.cl]), torch.tensor(self.data[idx + self.cl : idx + self.cl + self.hl])

train_loader = DataLoader(TimeSeriesDataset(tr_data, max_context, max_horizon), batch_size=32, shuffle=True)
optimizer = optim.AdamW(model.model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print(f"🏋️ Training LoRA with tr_data (Context: {max_context})...")
model.model.train()
start_train_lora = time.time()

for epoch in range(5): 
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # [단계 1] 64 배수 패딩
        curr_len = batch_x.shape[1]
        tgt_len = ((curr_len + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
        
        if curr_len != tgt_len:
            pad_len = tgt_len - curr_len
            padding = torch.zeros((batch_x.shape[0], pad_len), device=device)
            batch_x_padded = torch.cat([padding, batch_x], dim=1)
            masks = torch.ones_like(batch_x_padded).to(device)
            masks[:, :pad_len] = 0
        else:
            batch_x_padded = batch_x
            masks = torch.ones_like(batch_x_padded).to(device)

        # [단계 2] 데이터를 [Batch, Num_Patches, 64] 구조로 Reshape
        num_patches = tgt_len // PATCH_SIZE
        batch_x_input = batch_x_padded.view(batch_x.shape[0], num_patches, PATCH_SIZE)
        masks_input = masks.view(batch_x.shape[0], num_patches, PATCH_SIZE)

        optimizer.zero_grad()
        
        # [단계 3] 모델 호출
        outputs = model.model(batch_x_input, masks_input)
        
        # [단계 4] 출력 처리
        if isinstance(outputs, tuple): outputs = outputs[0]
        if outputs.ndim == 4: # 분위수 차원 평균
            outputs = outputs.mean(dim=-1) 
        
        # 전체 시퀀스 펼치기 및 마지막 구간 슬라이싱
        outputs = outputs.reshape(batch_x.shape[0], -1)
        outputs = outputs[:, -max_horizon:]
        
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.6f}")

train_time_lora = time.time() - start_train_lora
print(f"✅ LoRA Training Complete: {train_time_lora:.2f}s")

###########################################################################################################
# Prediction & Evaluation
###########################################################################################################

print("🚀 Predicting with LoRA Enhanced Model...")
model.model.eval()
start_inf_lora = time.time()
lora_preds, _ = sliding_window_forecast(model, te_data, max_context, max_horizon)
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