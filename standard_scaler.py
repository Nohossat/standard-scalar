import pandas as pd
import numpy as np

class StdScaler:
    def __init__(self):
        # initiate the class with means and standard deviation lists
        self.params_ = {
            'means' : None,
            'std_dev': None
        }
        
    def get_values(self, data):
        # transform data if it is a dataframe
        data_values = data
        
        if isinstance(data, pd.core.frame.DataFrame):
            data_values = data.values
            
        return data_values
        
    def fit(self, data):
        # get means and standard deviation per feature
        data_values = self.get_values(data)
            
        self.params_['means'] = np.mean(data_values, 0)
        self.params_['std_dev'] = np.std(data_values, 0)
        
    def transform(self, data):
        # if means list empty => fit method not called
        if self.params_['means'] is None :
            raise Exception('You must call the fit method before transforming the data')
            
        data_values = self.get_values(data)
        # get zscore for each value
        return (data_values - self.params_['means']) /  self.params_['std_dev']
    
    def fit_transform(self, data):
        # call methods necessary for standardisation
        self.fit(data)
        return self.transform(data)
    
    @property
    def mean_(self):
        return self.params_['means']
    
    @property
    def var_(self):
        return self.params_['std_dev']