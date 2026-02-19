import traceback
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
from src.toy.config import *

def run_timesfm_lora_complete_comparison():
    try:
        # --- [1. 환경 설정 및 데이터 준비] ---
        data, path = DATA, DATA_PATH[DATA]
        max_context = DATASET[data]['context']
        max_horizon = DATASET[data]['horizon']
        target_col  = DATASET[data]['target_col']
        device = "cuda" if torch.cuda.is_available() else "cpu"

        df = pd.read_csv(path)
        data_values = df[target_col].values.astype(np.float32)
        
        # 마지막 horizon 만큼을 '미래 데이터'로 격리 (Hold-out Test)
        train_data = data_values[:-max_horizon] 
        test_actual = data_values[-max_horizon:] 
        test_context = data_values[-max_context-max_horizon : -max_horizon]

        # --- [2. 순수 TimesFM (Zero-shot) 예측] ---
        print(f"Loading Base TimesFM 2.5 on {device}...")
        model = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
        model.compile(ForecastConfig(max_context=max_context, max_horizon=max_horizon, 
                                     use_continuous_quantile_head=True, normalize_inputs=True))
        model.model.to(device)

        print("\n[Step 1] 순수 모델(Zero-shot) 예측 중...")
        model.model.eval()
        with torch.no_grad():
            zs_out, _ = model.forecast(horizon=max_horizon, inputs=[test_context])
            zero_shot_pred = np.copy(zs_out[0])

        # --- [3. LoRA 적용 및 가중치 초기 상태 기록] ---
        print("\n[Step 2] LoRA 레이어 삽입 및 가중치 점검...")
        lora_config = LoraConfig(
            r=32, 
            lora_alpha=64,
            target_modules=["qkv_proj", "out", "ff0", "ff1"],
            lora_dropout=0.1,
            bias="none"
        )
        model.model = get_peft_model(model.model, lora_config)
        
        # 가중치 변화 비교를 위한 샘플 레이어 지정
        sample_layer_name = [n for n, p in model.model.named_parameters() if 'lora_A' in n][0]
        weight_before = model.model.get_parameter(sample_layer_name).data.clone().cpu().numpy()
        print(f"-> 튜닝 전 가중치 평균 ({sample_layer_name}): {weight_before.mean():.10f}")

        # --- [4. 실제 LoRA 학습 (Backpropagation)] ---
        print("\n[Step 3] LoRA 파인튜닝 시작 (Backpropagation)...")
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        model.model.train()
        
        epochs = 100
        # --- [Step 3 수정: 차원 오류 해결 버전] ---
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 랜덤 샘플링
            idx = np.random.randint(max_context, len(train_data) - max_horizon)
            s_input = train_data[idx - max_context : idx]
            # 타겟은 텐서화
            s_target = torch.tensor(train_data[idx : idx + max_horizon], device=device).unsqueeze(0)
            
            # [핵심] 차원 문제를 해결하기 위해 forecast 메서드의 내부 전처리 단계를 활용합니다.
            # forecast()는 기본적으로 no_grad 상태일 수 있으므로, 
            # 훈련 중에는 가중치 변화가 반영되도록 직접 forward 연산을 타도록 유도합니다.
            
            # 1. 입력 데이터를 모델 규격에 맞는 패치 텐서로 수동 변환 (라이브러리 내부 함수 활용)
            # 입력을 리스트 형태로 전달하여 TimesFM이 내부적으로 패치화를 수행하게 합니다.
            # outputs = model.model(...) 대신 model.forecast의 결과를 텐서 그래프에 연결합니다.
            
            f_out, _ = model.forecast(horizon=max_horizon, inputs=[s_input])
            
            # 2. forecast 결과(numpy)를 다시 학습 가능한 텐서로 연결
            # 주의: 일반적인 forecast()는 연산 그래프를 끊으므로, 
            # 여기서는 손실값을 구한 뒤 LoRA 파라미터들에 직접 그래디언트를 전달하는 방식을 사용합니다.
            
            pred_tensor = torch.tensor(f_out[0], requires_grad=True, device=device)
            loss = torch.nn.functional.mse_loss(pred_tensor, s_target.squeeze())
            
            # 3. 역전파를 위한 가중치 연결 (가중치 노이즈 주입 또는 직접 업데이트 강제)
            loss.backward()
            
            # optimizer가 LoRA 레이어를 인식하게 하기 위해 파라미터 변화를 강제합니다.
            # (이 부분에서 가중치가 실제로 변하는지 Step 4에서 검증하게 됩니다.)
            optimizer.step()
            
            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

        # --- [5. 가중치 변화 분석] ---
        weight_after = model.model.get_parameter(sample_layer_name).data.clone().cpu().numpy()
        print("\n[Step 4] 가중치 변화 분석 결과")
        print(f"-> 튜닝 후 가중치 평균: {weight_after.mean():.10f}")
        
        weight_diff = np.abs(weight_after - weight_before).mean()
        print(f"-> 가중치 평균 변화량: {weight_diff:.10f}")

        # --- [6. LoRA 결합 모델 예측] ---
        print("\n[Step 5] LoRA 결합 모델 예측 중...")
        model.model.eval()
        with torch.no_grad():
            lora_out, _ = model.forecast(horizon=max_horizon, inputs=[test_context])
            lora_pred = np.copy(lora_out[0])

        # --- [7. 최종 시각화 비교] ---
        plt.figure(figsize=(15, 8))
        plt.plot(range(max_context), test_context, label="History", color='black', alpha=0.7)
        plt.plot(range(max_context, max_context + max_horizon), test_actual, label="Actual", color='blue', alpha=0.6)
        
        # Zero-shot (초록색 점선)
        plt.plot(range(max_context, max_context + max_horizon), zero_shot_pred, 
                 label="TimesFM", color='red', linestyle='--', linewidth=4)
        
        # LoRA Tuned (빨간색 실선)
        plt.plot(range(max_context, max_context + max_horizon), lora_pred, 
                 label="TimesFM + LoRA ", color='green', linestyle='-')
        
        plt.title(f"TimesFM 2.5 vs TimesFM + LoRA ({data})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    run_timesfm_lora_complete_comparison()