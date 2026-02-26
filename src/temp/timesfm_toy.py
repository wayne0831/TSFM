import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
from config import *

def run_timesfm_full_comparison():
    print("🚀 Starting TimesFM 2.5 Full Data Comparison...")
    try:
        # 0. 설정 로드
        data_name   = DATA
        path        = DATA_PATH[data_name]
        max_context = 96 # DATASET[data_name]['context']
        max_horizon = 192 # DATASET[data_name]['horizon']
        target_col  = 'OT'# DATASET[data_name]['target_col']

        # 1. 모델 로드
        print(f"🚀 Loading TimesFM 2.5 200M (Version: {MODEL_VER})...")
        model = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
        
        config = ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            use_continuous_quantile_head=True, 
            normalize_inputs=True
        )
        model.compile(config)

        # 2. 데이터 로드
        print(f"📊 Loading {data_name} data...")
        df = pd.read_csv(path)
        data_values = df[target_col].values.astype(np.float32)
        
        # 3. 전체 데이터 구간 예측 (Sliding Window)
        # Context 구간 이후부터 예측을 시작합니다.
        total_len = len(data_values)
        all_predictions = np.full(total_len, np.nan) # 예측값이 없는 곳은 NaN 처리
        
        print("🔍 Performing full-range forecasting...")
        
        # max_horizon 간격으로 이동하며 예측 수행
        for start_idx in range(max_context, total_len, max_horizon):
            # 현재 시점 이전의 데이터를 context로 사용
            current_context = data_values[max(0, start_idx - max_context) : start_idx]
            
            # 예측할 남은 길이가 horizon보다 작을 수 있음
            current_horizon = min(max_horizon, total_len - start_idx)
            if current_horizon <= 0: break
            
            forecast_output, _ = model.forecast(
                horizon=current_horizon,
                inputs=[current_context],
            )
            
            # 결과 저장
            prediction = forecast_output[0]
            all_predictions[start_idx : start_idx + current_horizon] = prediction
            
            if start_idx % (max_horizon * 5) == 0:
                print(f"Progress: {start_idx}/{total_len} points processed...")

        # 4. 시각화
        plt.figure(figsize=(15, 7))
        
        # 실제 전체 데이터
        plt.plot(data_values, label="Actual (True)", color='black', alpha=0.4, linewidth=1)
        
        # TimesFM 예측 데이터 (NaN 구간 제외하고 출력됨)
        plt.plot(all_predictions, label="TimesFM 2.5 Prediction", color='red', linestyle='--', linewidth=1.2)
        
        plt.axvline(x=max_context, color='blue', linestyle=':', label='Forecast Start')
        
        plt.title(f"TimesFM 2.5: Full Data Prediction vs Actual ({data_name} - {target_col})")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # 성능 지표 간단 계산 (예측값이 존재하는 구간만)
        valid_idx = ~np.isnan(all_predictions)
        mse = np.mean((data_values[valid_idx] - all_predictions[valid_idx])**2)
        print(f"✅ Forecast Complete. Mean Squared Error: {mse:.4f}")
        
        plt.show()

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_timesfm_full_comparison()