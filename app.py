import pandas as pd
import numpy as np
from standard_scaler import StdScaler

X = np.array([[1, 20, 6], [4, 35, 10], [7, 50, 18]])
df_X = pd.DataFrame(X)

# 2 cases : 

scaler = StdScaler()
# standardize a numpy array
scaled_X = scaler.fit_transform(X)
print(scaler.mean_, scaler.var_)


scaler_df = StdScaler()
# standardize a pandas Dataframe
scaler_df.fit(df_X)
print(scaler_df.mean_, scaler_df.var_)

scaled_df = scaler_df.transform(df_X)

