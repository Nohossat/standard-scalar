import pandas as pd
import numpy as np
from standard_scaler import StdScaler

df_X = pd.read_csv('vine.csv')
X = df_X.values

scaler = StdScaler()

# 2 cases : 

# standardize a pandas Dataframe
scaler.fit_transform(df_X)
print(scaler.mean_, scaler.var_)


# standardize a numpy array
scaled_X = scaler.fit_transform(X)
print(scaler.mean_, scaler.var_)
