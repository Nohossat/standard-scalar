# Standard Scaler

A custom implementation of Standard Scaler in Python.

The object Standard Scaler is based on the scikit-learn implementation.

## Usage

You can either standardize a Pandas Dataframe or a Numpy array.

```python

import pandas as pd
import numpy as np
from standard_scaler import StdScaler

scaler = StdScaler()
df_X = pd.read_csv('vine.csv')
X = df_X.values

# standardize a Pandas dataframe
scaled_df_X = scaler.fit_transform(df_X)

# standardize numpy array
scaler = StdScaler()
scaled_X = scaler.fit_transform(X)
```