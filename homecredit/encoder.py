import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from homecredit.data import HomeCredit
from homecredit.preparation import Preparation
from homecredit.cleaner import Cleaning
from homecredit.exploration import Exploration


class Encoder():
    
    """ Initialize dataframe
    """
    def __init__(self):

        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)
        # Assign an attribute ".data" to all new instances of Preparation
        self.data = Cleaning().remove_missvalues() # good practice to be sure not to modify your `data` variable
        self.catcols = Preparation().get_catcols()
        self.numcols = Preparation().get_numcols() 
        
        
    def execute(self):
        
        copy = self.data.copy()
        
        for col_name in self.catcols:
    
            L = list(copy[col_name].unique())
            if '' in L:
                copy[col_name].replace("", "NoValue", inplace=True) #Replace NaN by "NoCodeNature"

            ohe = OneHotEncoder(sparse = False) # Instanciate encoder
            ohe.fit(copy[[col_name]]) # Fit encoder  ---> OneHotEncoder(sparse=False)

            col_encoded = ohe.transform(copy[[col_name]]) # Encode

            dicts_col = {}
            keys = list(ohe.categories_[0])
            values = col_encoded.T.astype(int)

            for i,j in enumerate(keys):
                dicts_col[j] = values[i,:]

            result = pd.DataFrame.from_dict(dicts_col)

            copy = copy.reset_index(drop=True)

            #Concat self.data and result dataframes
            data_res = pd.concat([copy, result], axis = 1)

            if 'NoValue' in list(data_res.columns):
                data_res = data_res.drop(columns= ['NoValue',col_name] )
                copy = data_res
            else:
                data_res = data_res.drop(columns= col_name)
                copy = data_res
        
        encoded_df = copy

        return encoded_df
        
 