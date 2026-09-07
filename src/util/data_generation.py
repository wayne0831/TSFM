# import os
# import sys

# # 현재 파일(src/util/data_generation.py) 기준 2단계 상위 폴더(프로젝트 루트: TSFM_Ops)를 sys.path에 등록
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from PyEMD import EMD
from typing import List, Dict, Any, Tuple
import warnings
import os

from src.config import *
from src.util.data_analysis import calculate_time_series_strength, estimate_period_fft
import ast
import time

warnings.filterwarnings('ignore')

# =====================================================================
# 3. KernelSynth 커널 클래스 정의
# =====================================================================
class Kernel:
    def __init__(self, name: str, params: Dict[str, Any], expr: str):
        self.name, self.params, self.expr = name, params, expr
    def __call__(self, x1, x2): raise NotImplementedError
    def __add__(self, other): return CombinedKernel(self, other, op="+")
    def __mul__(self, other): return CombinedKernel(self, other, op="*")

class CombinedKernel(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel, op: str):
        self.k1, self.k2, self.op = k1, k2, op
        super().__init__(name="Combined", params={}, expr=f"({k1.expr}{op}{k2.expr})")
    def __call__(self, x1, x2):
        if self.op == "+": return self.k1(x1, x2) + self.k2(x1, x2)
        elif self.op == "*": return self.k1(x1, x2) * self.k2(x1, x2)

class ConstantKernel(Kernel):
    def __init__(self, c=1.0):
        super().__init__("CONST", {"c": c}, f"CONST(c={c:g})")
        self.c = c
    def __call__(self, x1, x2): return np.full((len(x1), len(x2)), self.c)

class WhiteNoiseKernel(Kernel):
    def __init__(self, sigma_n):
        super().__init__("WN", {"sigma_n": sigma_n}, f"WN(sigma={sigma_n:g})")
        self.sigma_n = sigma_n
    def __call__(self, x1, x2): 
        return np.where(np.abs(x1[:, None] - x2[None, :]) < 1e-6, self.sigma_n, 0.0)

class LinearKernel(Kernel):
    def __init__(self, sigma):
        super().__init__("LIN", {"sigma": sigma}, f"LIN(sigma={sigma:g})")
        self.sigma = sigma
    def __call__(self, x1, x2): return (self.sigma**2) + np.outer(x1, x2)

class RBFKernel(Kernel):
    def __init__(self, length_scale):
        super().__init__("RBF", {"length_scale": length_scale}, f"RBF(l={length_scale:g})")
        self.l = length_scale
    def __call__(self, x1, x2): 
        return np.exp(-((x1[:, None] - x2[None, :])**2) / (2.0 * (self.l**2)))

class RationalQuadraticKernel(Kernel):
    def __init__(self, alpha, c=1.0):
        super().__init__("RQ", {"alpha": alpha, "c": c}, f"RQ(alpha={alpha:g})")
        self.alpha, self.c = alpha, c
    def __call__(self, x1, x2): 
        return (1.0 + ((x1[:, None] - x2[None, :])**2) / (2.0 * self.alpha))**(-self.c)

class PeriodicKernel(Kernel):
    def __init__(self, period):
        super().__init__("PER", {"period": period}, f"PER(p={period:g})")
        self.p = period
    def __call__(self, x1, x2): 
        diff = np.abs(x1[:, None] - x2[None, :])
        return np.exp(-2.0 * (np.sin(np.pi * diff / self.p)**2))

def build_kernel_bank() -> List[Kernel]:
    bank = [ConstantKernel(c=1.0)]
    for s_n in [0.1, 1.0]: bank.append(WhiteNoiseKernel(sigma_n=s_n))
    for s in [0.0, 1.0, 10.0]: bank.append(LinearKernel(sigma=s))
    for l in [0.1, 1.0, 10.0]: bank.append(RBFKernel(length_scale=l))
    for alpha in [0.1, 1.0, 10.0]: bank.append(RationalQuadraticKernel(alpha=alpha))
    periods = [24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12, 40, 10]
    for p in periods: bank.append(PeriodicKernel(period=p))
    return bank

# =====================================================================
# 4. 시계열 합성 함수 (라벨 할당 배제)
# =====================================================================
def kernel_synth_generate(kernel_bank, max_kernels=5, length=512, jitter=1e-5):
    j = np.random.randint(1, max_kernels + 1)
    selected_kernels = [kernel_bank[idx] for idx in np.random.choice(len(kernel_bank), size=j, replace=True)]
    
    composed_kernel = selected_kernels[0]
    kernels_used = [selected_kernels[0].name]
    operations = []

    for i in range(1, j):
        op = np.random.choice(["+", "*"])
        operations.append(op)
        kernels_used.append(selected_kernels[i].name)
        if op == "+": composed_kernel = composed_kernel + selected_kernels[i]
        else: composed_kernel = composed_kernel * selected_kernels[i]

    t = np.linspace(0, length - 1, length)
    cov_matrix = composed_kernel(t, t) + np.eye(length) * jitter
    synthetic_series = np.random.multivariate_normal(np.zeros(length), cov_matrix)
    
    meta = {
        "Num_Kernels": j,
        "Kernel_Expression": composed_kernel.expr,
        "Operations": operations if operations else ["None"],
        "Kernels_Used": kernels_used
    }
    return synthetic_series, meta

def assign_sector_label(kernels_used: List[str]) -> Tuple[str, str]:
    """커널 구성 요소에 따라 S1~S4 정답(Ground Truth) 섹터를 추론합니다."""
    has_per = any("Periodic" in k for k in kernels_used)
    has_lin = any("Linear" in k for k in kernels_used)
    
    if has_per and has_lin:
        return "S1", "Composite"
    elif has_per and not has_lin:
        return "S2", "Seasonal"
    elif not has_per and has_lin:
        return "S4", "Trending"
    else:
        return "S3", "Stationary"


# =====================================================================
# 5. 실행: 데이터 생성 및 Null 컬럼 포함 CSV 저장
# =====================================================================
if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(42)

    num_samples = PARAMS[DATA_GEN_METHOD]["NUM_SAMPLES"]
    length      = PARAMS[DATA_GEN_METHOD]["LENGTH"]
    meta_path   = RES_PATH['data_generation'][DATA_GEN_METHOD]['METADATA']
    data_path   = RES_PATH['data_generation'][DATA_GEN_METHOD]['DATA']

    bank = build_kernel_bank()
    records = []
    ts_list = []  # 시계열 배열을 수집할 리스트

    print(f"💡 KernelSynth 기반 시계열 데이터 {num_samples}개 생성 및 지표 산출")

    for i in range(num_samples):
        ts, meta = kernel_synth_generate(bank, max_kernels=5, length=length)
        ts_scaled = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-9)
        # 2차원 배열 저장을 위해 float32 변환 후 리스트에 추가 (ts_scaled 또는 ts 원본 중 선택 가능)
        ts_list.append(ts_scaled.astype(np.float32))

        ft_stl, fs_stl, fr_stl, ft_emd, fs_emd, fi_emd = calculate_time_series_strength(ts_scaled)
        
        # True_Sector와 Pattern은 None(NaN)으로 적재
        record = {
            "True_Sector": None,
            "Pattern": None,
            "F_T_STL": ft_stl,
            "F_S_STL": fs_stl,
            "F_R_STL": fr_stl,
            "F_T_STL_EMD": ft_emd,
            "F_S_STL_EMD": fs_emd,
            "F_I_STL_EMD": fi_emd,
            "Num_Kernels": meta["Num_Kernels"],
            "Kernel_Expression": meta["Kernel_Expression"],
            "Operations": str(meta["Operations"]),
            "Kernels_Used": str(meta["Kernels_Used"])  # 후속 라벨링을 위한 메타데이터 보존
        }
        records.append(record)
        
        if (i + 1) % 100 == 0:
            print(f"  - {i + 1}/{num_samples} 진행 완료")

    df_raw = pd.DataFrame(records)
    
    columns_order = [
        "True_Sector", "Pattern", 
        "F_T_STL", "F_S_STL", "F_R_STL", 
        "F_T_STL_EMD", "F_S_STL_EMD", "F_I_STL_EMD", 
        "Num_Kernels", "Kernel_Expression", "Operations", "Kernels_Used"
    ]

    df_raw = df_raw[columns_order]

    # 메타 데이터 저장
    df_raw.to_csv(meta_path, index=False)
    print(f"\n✅ {DATA_GEN_METHOD} 기반 시계열 데이터 생성 완료 {meta_path}")

    # 생성된 시계열 저장
    ts_array = np.vstack(ts_list)  # 또는 np.array(ts_list, dtype=np.float32)
    npz_path = os.path.splitext(data_path)[0] + ".npz"
    np.savez_compressed(npz_path, time_series=ts_array)
    print(f"✅ {DATA_GEN_METHOD} 시계열 e데이터 저장 완료: {npz_path} (Shape: {ts_array.shape})")


    print("\n💡 Sector 할당 로직 기반 시계열 데이터 sector 부여")
    df = pd.read_csv(meta_path)

    # 문자열 형태로 저장된 Kernels_Used 파싱 및 라벨 부여
    sectors = []
    patterns = []

    for _, row in df.iterrows():
        try:
            k_list = ast.literal_eval(row["Kernels_Used"])
        except Exception:
            k_list = [str(row["Kernels_Used"])]
            
        expr = str(row["Kernel_Expression"])
        sec, pat = assign_sector_label(kernels_used=k_list)
        sectors.append(sec)
        patterns.append(pat)

    # 컬럼 채우기
    df["True_Sector"] = sectors
    df["Pattern"] = patterns

    # 최종 저장 (Kernels_Used 컬럼은 필요 시 drop=True 가능)
    df.to_csv(meta_path, index=False)

    print("✅ 시계열 데이터 sector 할당 완료")
    print("\n[할당된 섹터별 데이터 분포]")
    print(df["True_Sector"].value_counts())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n⏱️ 총 소요 시간: {elapsed_time/60:.2f}분")