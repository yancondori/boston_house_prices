import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import seaborn as sns

boston_hp = pd.read_csv('boston.csv')
# print(boston_hp.head())

# def cap_outliers(series, lower_quantile=0.2, upper_quantile=0.85):
#     lower_bound = series.quantile(lower_quantile)
#     upper_bound = series.quantile(uppe    r_quantile)
#     return series.apply(lambda x: max(min(x, upper_bound), lower_bound))

# capped_df = boston_hp.copy()
# for col in capped_df.columns:
#     capped_df[col] = cap_outliers(capped_df[col])


capped_df = boston_hp.copy()
for col in capped_df.columns:
    capped_df[col] = winsorize(capped_df[col], limits=[0.01, 0.01])

# plt.figure(figsize=(10, 20))

# Hacemos un Loop a traves del las columnas del DataFrame para crear boxplot de cada una de las variables
for i, col in enumerate(capped_df.columns):
    plt.subplot(len(capped_df.columns), 1, i+1)
    sns.boxplot(x=capped_df[col], orient='h')
    plt.title(f'Boxplot de {col}')
    plt.tight_layout()
    plt.show() 

print(capped_df.head())

# >>> a = np.array([10, 4, 9, 8, 5, 3, 7, 2, 1, 6])
# The 10% of the lowest value (i.e., 1) and the 20% of the highest values (i.e., 9 and 10) are replaced.

# >>> winsorize(a, limits=[0.1, 0.2])