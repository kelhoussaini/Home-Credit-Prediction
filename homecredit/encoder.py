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

import pickle

class Encoder:
    
    """ Initialize dataframe
    """
    def __init__(self, data_set = 'train', cols = None, newdf = None, targ= "TARGET"):
        
        # cols : list that includes selected features & target variable
        # with cols, we reduce the dataframe columns 
        # and use it specially for depoying api

        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)
        # Assign an attribute ".data" to all new instances of Preparation

         # Preparation
        self.prep = Preparation(data_set, cols)
        
        # Cleaning
        self.cl = Cleaning('train', cols, newdf, targ) 
        
        #print("cols", list(self.cl.data.columns))
        
        self.cl.prep.data_set = data_set
                
        self.data = self.cl.remove_missvalues(set_df = 'train') 
        
        #print("list(self.data.columns) ", list(self.data.columns))
        
        # cleaning new data to predict
        
        if newdf is None: # data to predict does not exist 
            if cols is None: # && we consider all features
                #print("here0 : --->", self.data.columns)
                T = list(self.data.columns)[:]
                #print("yes", T)
                #T.remove(targ)
                #self.tt = Cleaning('test', T)
            else:   # selected features         
                #print("here1")
                T = cols[:] #  [:] to keep the original list 'cols' unchanged
                #print("T    ", T)
            #T.remove(targ)
            #print("here2", T)
            self.tt = Cleaning('train', T, newdf, targ)
            self.new_data = self.tt.remove_missvalues(set_df = 'test') 
        else:
            #print("here2")
            self.nn = Cleaning('train', cols, newdf)
            #print("here2))", self.nn.__dict__.keys(), self.nn.data.shape, self.nn.newdata.shape)
            self.new_data = self.nn.remove_missvalues(set_df = 'test') 
            #print("here20", list(self.new_data.columns))
            
        
        ##############  N.B  #############
        #for the data to predict, we only make transformation
        # without fitting, for this reason, we create a new instance of Cleaning() for test dataset
        ##############  N.B (END) #############
                
                 
        
    def execute(self, data_topredict=False): #new_data : we also need to encode/transform any data we want to predict 
        
        df = self.data
        df_new = self.new_data
        
        catcols = self.prep.get_catcols()
        
        results = []
        for col_name in catcols:
        
            #print(" ***** ")
            L = list(df[col_name].unique())
            

            #print("L :   ", L)
            if '' in L:
                df[col_name]=df[col_name].replace("", "NoValue") #Replace NaN by "NoValue"

            ohe = OneHotEncoder(sparse = False) # Instanciate encoder
            col_encoded = ohe.fit_transform(df[[col_name]]) # Encode

            dicts_col = {}
            keys = list(ohe.categories_[0])
            values = col_encoded.T.astype(int)

            for i,j in enumerate(keys):
                dicts_col[j] = values[i,:]
                
            result = pd.DataFrame.from_dict(dicts_col)
            df = df.reset_index(drop=True)
            #Concat df and result dataframes
            data_res = pd.concat([df, result], axis = 1)

            if 'NoValue' in list(data_res.columns):
                data_res = data_res.drop(columns= ['NoValue',col_name] )
                df = data_res
            else:
                data_res = data_res.drop(columns= col_name)
                df = data_res

            if data_topredict:
                L_test = list(df_new[col_name].unique())
                if '' in L_test:
                    df_new[col_name]=df_new[col_name].replace("", "NoValue")

                ##############  N.B  #############
                #for the data to predict, we only make transformation
                # without fitting, for this reason, we add above these lines
                ##############  N.B (END) #############
                
                new_encoded = ohe.transform(df_new[[col_name]]) # transforming data to predict
                dicts_newdata = {}
                values_newdata = new_encoded.T.astype(int)
                for i,j in enumerate(keys):
                    dicts_newdata[j] = values_newdata[i,:]

                result_newdata = pd.DataFrame.from_dict(dicts_newdata)
                df_new = df_new.reset_index(drop=True)
                data_res_new = pd.concat([df_new, result_newdata], axis = 1)

                if 'NoValue' in list(data_res_new.columns):
                    data_res_new = data_res_new.drop(columns= ['NoValue',col_name] )
                    df_new = data_res_new
                else:
                    data_res_new = data_res_new.drop(columns= col_name)
                    df_new = data_res_new
                    
                    
            results.append({col_name : ohe})
            
        with open("encoder.pckl", "wb") as file:
            pickle.dump(results, file) 
        
        return (df, df_new) if data_topredict else df


        
 