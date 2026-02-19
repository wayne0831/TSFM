###########################################################################################################
# import libraries
###########################################################################################################

from src.toy.config import * 
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

###########################################################################################################
# run code
###########################################################################################################

def run_timesfm_2p5_etth():

    try:
        data        = DATA
        path        = DATA_PATH[data]
        max_context = DATASET[data]['context']
        max_horizon = DATASET[data]['horizon']
        target_col  = DATASET[data]['target_col']

        # 1. 모델 로드
        print("Loading TimesFM 2.5 200M PyTorch model...")
        model = TimesFM_2p5_200M_torch.from_pretrained(MODEL_VER)
        
        config = ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            use_continuous_quantile_head=True, 
            normalize_inputs=True
        )
        model.compile(config)

        # 2. ETTh1 데이터 직접 로드 (GitHub의 CSV 원본 주소 사용)
        print("Loading ETTh1 data from GitHub...")
        df = pd.read_csv(path)
        
        # 'OT' (Oil Temperature) 컬럼 사용
        data_values = df[target_col].values.astype(np.float32)
        
        # 전체 데이터 중 마지막 부분을 테스트로 사용
        context = data_values[-max_context-max_horizon : -max_horizon]
        actual = data_values[-max_horizon:] 
        # 3. 예측 수행
        print("Performing forecast...")
        forecast_output, _ = model.forecast(
            horizon=max_horizon,
            inputs=[context],
        )
        
        prediction = forecast_output[0]

        # 4. 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(context)), context, label="History", color='black', alpha=0.7)
        plt.plot(range(len(context), len(context) + len(actual)), actual, label="Actual", color='blue', alpha=0.6)
        plt.plot(range(len(context), len(context) + len(prediction)), prediction, label="TimesFM 2.5", color='red', linestyle='--')
        plt.title("TimesFM 2.5 Zero-shot Forecast on ETTh1 (OT)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_timesfm_2p5_etth()