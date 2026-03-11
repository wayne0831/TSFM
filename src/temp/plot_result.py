import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    print(os.getcwd())

    # 1. 데이터 및 예측 결과 로드
    df = pd.read_csv('./data/ETTh1.csv')
    target = df['OT'].values  # 실제 OT 컬럼

    # 파일명은 사용자 환경에 맞춰 확인 필요
    preds_base = np.load('./results/predictions/TimesFM/TimesFM_Etth1_cl96_hl96_preds.npy')
    preds_lora = np.load("./results/predictions/LoRA/TimesFM_Etth1_cl96_hl96_LoRA_r2_a8_d0.1_tgt['ff0', 'ff1']_lr0.0001_e10_bs16_preds.npy")

    # 2. 테스트 세트 구간 설정 (0.7 비율)
    # 실험 코드의 ft_len = int(len(target) * 0.7) 로직을 따름
    ft_len = int(len(target) * 0.7)

    # 예측값의 길이에 맞춰 실제값(Ground Truth) 슬라이싱
    # te_data = target[ft_len - 96:] 였고, forecast가 96(cl) 이후부터 예측하므로 
    # 실제 예측의 시작점은 target[ft_len] 임
    actuals = target[ft_len : ft_len + len(preds_base)]

    # 3. 그래프 구현
    plt.figure(figsize=(15, 6))

    # 실제값 (검정 실선)
    plt.plot(actuals, label='Actual (OT)', color='black', alpha=0.5, linewidth=1.2)
    # 기본 TimesFM 예측 (파란 점선)
    plt.plot(preds_base, label='TimesFM Prediction', color='blue', linestyle='--', alpha=0.8)
    # LoRA 튜닝 후 예측 (빨간 점선)
    plt.plot(preds_lora, label='TimesFM + LoRA Prediction', color='red', linestyle=':', alpha=0.8)

    plt.title('TimesFM vs TimesFM + LoRA Forecasting Comparison (ETTh1 OT)')
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Oil Temperature (OT)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # 전체 결과 저장
    plt.savefig('forecast_comparison_full.png')

    # 가독성을 위해 앞부분 500개만 확대해서 별도 저장
    plt.xlim(0, 500)
    plt.title('Forecasting Comparison (Zoomed - First 500 steps)')
    plt.savefig('forecast_comparison_zoom.png')

    print(f"Total test steps: {len(actuals)}")