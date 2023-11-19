import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import seaborn as sns

boston_hp = pd.read_csv('boston.csv')

from scipy.stats.mstats import winsorize
boston_cap = boston_hp.copy()

# Apply winsorizing to each column and update the boston_capped
boston_cap['CRIM'] = winsorize(boston_hp['CRIM'], limits=[0.00, 0.15])
boston_cap['ZN'] = winsorize(boston_hp['ZN'], limits=[0.0, 0.15])
boston_cap['CHAS'] = winsorize(boston_hp['CHAS'], limits=[0.0, 0.1])
boston_cap['RM'] = winsorize(boston_hp['RM'], limits=[0.1, 0.1])
boston_cap['DIS'] = winsorize(boston_hp['DIS'], limits=[0.0, 0.1])
boston_cap['PTRATIO'] = winsorize(boston_hp['PTRATIO'], limits=[0.1, 0.0])
boston_cap['B'] = winsorize(boston_hp['B'], limits=[0.20, 0.0])
boston_cap['LSTAT'] = winsorize(boston_hp['LSTAT'], limits=[0.0, 0.1])
boston_cap['MEDV'] = winsorize(boston_hp['MEDV'], limits=[0.1, 0.1])

# differences = (boston_cap != boston_hp)

# # Create a DataFrame that shows only the entries that have been changed
# changed_entries = differences.apply(lambda x: x.index[x].tolist(), axis=1)

# # Display the changed entries
# print(changed_entries)

# boston_cap.to_csv('boston_cap.csv',index=False)

print(boston_hp.describe())
print(boston_cap.describe())


