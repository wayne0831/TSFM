import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_kernel_features(df: pd.DataFrame) -> pd.DataFrame:
  """Kernel_Expression 및 메타 컬럼으로부터 상관관계 분석용 수치형 특성을 추출합니다."""
  features = pd.DataFrame(index=df.index)

  # 1. 6대 기본 커널 태그 매핑 (RQ 버그 수정 반영)
  kernel_tag_map = {
      'LIN': 'LIN',
      'PER': 'PER',
      'RBF': 'RBF',
      'RQ': 'RQ',
      'WN': 'WN',
      'CONST': 'CONST',
  }
  expr_series = df['Kernel_Expression'].astype(str)

  for k, tag in kernel_tag_map.items():
    features[f'has_{k}'] = (
        expr_series.str.contains(tag, regex=False)
    ).astype(int)
    features[f'cnt_{k}'] = expr_series.apply(lambda s: s.count(tag))

  # 2. 연산자 특성 대칭 구성 (has_add, cnt_add, has_mul, cnt_mul)
  features['cnt_add'] = expr_series.apply(lambda s: s.count('+'))
  features['has_add'] = (features['cnt_add'] > 0).astype(int)

  features['cnt_mul'] = expr_series.apply(lambda s: s.count('*'))
  features['has_mul'] = (features['cnt_mul'] > 0).astype(int)

  # 3. 전체 커널 복잡도 (Num_Kernels)
  if 'Num_Kernels' in df.columns:
    features['Num_Kernels'] = df['Num_Kernels']
  else:
    features['Num_Kernels'] = features['cnt_mul'] + features['cnt_add'] + 1

  return features


# =====================================================================
# 실행 및 상관계수 매트릭스 도출
# =====================================================================
file_path = 'C:/Users/AICT/Desktop/PythonProject/TSFM_Ops/results/data_generation/260905_KernelSynth_Num5000_Len512.csv'
df = pd.read_csv(file_path)

strength_metrics = [
    'F_T_STL',
    'F_S_STL',
    #'F_T_STL_EMD',
    #'F_S_STL_EMD',
    'F_I_STL_EMD',
]
X_kernel = parse_kernel_features(df)
df_corr_target = pd.concat([X_kernel, df[strength_metrics]], axis=1)

# 피어슨 상관계수 산출
corr_matrix = df_corr_target.corr(method='pearson')
cross_corr = corr_matrix.loc[X_kernel.columns, strength_metrics]

print('=== 커널 구성 특성과 분해 지표 간 피어슨 상관계수 ===')
print(cross_corr.round(4))

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(
    cross_corr,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.3f',
    cbar_kws={'label': 'Pearson Correlation'},
)
plt.title(
    'Correlation between Kernel Compositions and Time Series Strength Metrics',
    fontsize=13,
    fontweight='bold',
)
plt.xlabel('Strength Metrics', fontsize=11, fontweight='bold')
plt.ylabel('Kernel Architectural Features', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()