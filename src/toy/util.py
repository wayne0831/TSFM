###########################################################################################################
# import libraries
###########################################################################################################

import numpy as np
import torch

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