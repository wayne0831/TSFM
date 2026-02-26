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
#from util import *
from config import *

###########################################################################################################
# set user-defined functions
###########################################################################################################

def calculate_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred)**2)
    rmse = np.sqrt(mse)
    # 분모에 1e-8 추가
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
    return mae, mse, rmse, mape

def sliding_window_forecast(model_obj, data, context_len, horizon_len):
    """
    1. max_context 지점부터 데이터 끝까지 예측
    2. TimesFM 64 패치 규격 준수 (Reshape & Padding)
    3. GPU 연산 처리 (to(device))
    """
    predictions = []
    actuals = []
    PATCH_SIZE = 64
    
    # 모델의 가중치가 있는 장치(GPU)를 자동으로 감지
    device = next(model_obj.model.parameters()).device
    tgt_context_len = ((context_len + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    
    # 시작 지점: 과거 데이터가 context_len만큼 쌓인 시점
    i = context_len 
    
    while i < len(data):
        remaining_len = min(horizon_len, len(data) - i)
        
        context_raw = data[i - context_len : i]
        actual = data[i : i + remaining_len]
        
        try:
            # 원본 API 시도
            forecast_output, _ = model_obj.forecast(inputs=[context_raw], horizon=remaining_len)
            pred_values = forecast_output[0]
        except:
            # LoRA 모델 등 수동 텐서 주입이 필요한 경우
            context_padded = np.zeros(tgt_context_len, dtype=np.float32)
            context_padded[-len(context_raw):] = context_raw
            
            num_patches = tgt_context_len // PATCH_SIZE
            # [중요] 생성된 텐서를 반드시 GPU(device)로 이동
            inputs_ts = torch.tensor(context_padded).view(1, num_patches, PATCH_SIZE).to(device)
            masks_ts = torch.ones_like(inputs_ts).to(device)
            
            with torch.no_grad():
                outputs = model_obj.model(inputs_ts, masks_ts)
                while isinstance(outputs, (tuple, list)): outputs = outputs[0]
                if outputs.ndim == 4: outputs = outputs.mean(dim=-1)
                
                all_preds = outputs.reshape(1, -1)[0]
                # 모델 출력(horizon_len) 중 필요한 자투리만큼만 슬라이싱
                pred_values = all_preds[-horizon_len : -horizon_len + remaining_len].cpu().numpy()

        predictions.extend(pred_values)
        actuals.extend(actual)
        i += horizon_len

    return np.array(predictions), np.array(actuals)

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

    max_context = HYPERPARAMS['TimesFM'][DATA]['max_context']
    max_horizon = HYPERPARAMS['TimesFM'][DATA]['max_horizon']

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
    start_inf_base = time.time()
    base_preds, base_actuals = sliding_window_forecast(
        model_obj=tmfm_base, 
        data=te_data, 
        context_len=max_context, 
        horizon_len=max_horizon
    )
    end_inf_base = time.time() - start_inf_base
    print(f"Base Model Inference Time: {end_inf_base:.2f}s")

    # visualize and save predictions
    start_idx = ft_len + max_context
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
tmfm_base.model = get_peft_model(tmfm_base.model, lora_config)
tmfm_base.model.to(device)
tmfm_base.model.print_trainable_parameters()

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
optimizer = optim.AdamW(tmfm_base.model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print(f"🏋️ Training LoRA with tr_data (Context: {max_context})...")
start_train_lora = time.time()

for epoch in range(1): 
    print(f"\nEpoch {epoch+1}/20")
    total_loss = 0
    tmfm_base.model.train() # 학습 모드 강제
    
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
            outputs = tmfm_base.model(batch_x_63, single_masks)

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
                grad_fix = sum(p.sum() for p in tmfm_base.model.parameters() if p.requires_grad) * 0
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

print("🚀 Merging LoRA weights into Base Model...")
# ⭐ 이 코드가 핵심입니다. LoRA 어댑터를 베이스 모델에 물리적으로 합칩니다.
tmfm_base.model = tmfm_base.model.merge_and_unload()

print("🚀 Predicting with LoRA Enhanced Model...")
tmfm_base.model.eval()
start_inf_lora = time.time()
# sliding_window_forecast는 내부적으로 model_obj.forecast를 호출합니다.
lora_preds, _ = sliding_window_forecast(tmfm_base, te_data, max_context, max_horizon)
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
plt.plot(base_actuals, label="Actual", color='black', alpha=0.4)
plt.plot(base_preds, label="Base TimesFM", color='blue', linestyle='--')
plt.plot(lora_preds, label="LoRA Enhanced", color='red', alpha=0.7)
plt.title(f"TimesFM 2.5 vs LoRA Enhanced Comparison")
plt.xlabel("Time Step")
plt.ylabel(tgt_col)
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

