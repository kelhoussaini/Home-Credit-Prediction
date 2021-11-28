import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from homecredit.data import HomeCredit
from homecredit.preparation import Preparation


    
class Cleaning:
    
    def __init__(self, data_set = 'train'):
        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)
        
        self.prep = Preparation()
        
        self.prep.data_set = data_set
        
        # Assign an attribute ".data" to all new instances of Preparation
        self.data = HomeCredit().get_data()#['train'].copy() # good practice to be sure not to modify your `data` variable
        
        #self.catcols = Preparation(self.prep.data_set).get_catcols()
        #self.numcols = Preparation(self.prep.data_set).get_numcols()
               
        
    def get_count_missvalues(self):
        
        df = self.data[self.prep.data_set].copy()
        
        missing_df = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
        return missing_df
    
    
    def get_percentage_missvalues(self):
        
        df = self.data[self.prep.data_set].copy()
        
        ratio = pd.DataFrame(
            (df.isnull().sum().sort_values(ascending=False))/ df.shape[0])
        return ratio
    
    
    def plot_missvalues_table(self, na_name=False): # self : dataframe
                                            # if na_names: print the features list  
    
        df = self.data[self.prep.data_set].copy()
        
        na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

        count = df[na_cols].isnull().sum().sort_values(ascending=False)
        ratio = ( df[na_cols].isnull().sum() / df.shape[0] * 100 ).sort_values(ascending=False)
        missing_df = pd.concat([count, np.round(ratio, 2)], axis=1, keys=['Number of missing values', 'Percent'])

        f,ax =plt.subplots(figsize=(19, 10))
        plt.xticks(rotation='90')
        fig=sns.barplot(missing_df.index, missing_df["Percent"])
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15);

        return na_cols if na_name else missing_df # na_cols : list, missing_df : dataframe
    
    
    # Remove entries 
    def remove_entries(self):
        
        df = self.data[self.prep.data_set]#.copy()
        
        df = df[df['CODE_GENDER'] != 'XNA'] # with gender = XNA
        df = df[df['NAME_FAMILY_STATUS'] != 'Unknown'] # 'Unknown' status
        # Remove entries with DAYS_EMPLOYED > 200_000
        df = df[df['DAYS_EMPLOYED'] < 200_000]
        df = df[df['AMT_ANNUITY'] < 150_000]
        df = df[df['AMT_GOODS_PRICE'] < 2.5* 10**6]  
        #self.data = self.data[self.data['OWN_CAR_AGE'] < 45]
        # when removing data with CAR_AGE > 45, we got NaN value :
        # cramers_val(self, col1, col2, margins=False), col1: FLAG_OWN_CAR and col2:all columns
        return df
    
    def remove_missvalues(self):
        
        #df = self.data[self.prep.data_set].copy()
        
        
        df1 = self.remove_entries()
        
        catcols = self.prep.get_catcols()
        numcols = self.prep.get_numcols()
        
        # Categorical Variables
        df1[catcols] = self.remove_entries()[catcols].replace(np.nan, '', regex=True)
        
        #self.remove_entries()[catcols].fillna("", inplace=True)

        # Replace the NaNs in numerical column by the mean of values
        # in numerical column respectively
        df1[numcols] = df1[numcols].fillna(value=df1[numcols].mean())
        # Also, we can remove these missing values
        return df1

        
        
