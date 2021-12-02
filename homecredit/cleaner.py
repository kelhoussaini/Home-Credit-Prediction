import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from homecredit.data import HomeCredit
from homecredit.preparation import Preparation


    
class Cleaning:
    
    def __init__(self, data_set = 'train', cols = None, newdf = None, targ= "TARGET"):
        # cols : list that includes selected features & target variable
        # with cols, we reduce the dataframe columns 
        # and use it specially for depoying api
        
        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)
        
        if data_set == 'train':
            self.prep = Preparation(data_set, cols)
        else:
            T = cols[:] #  [:] to keep the original list 'cols' unchanged
            T.remove(targ)
            self.prep = Preparation(data_set, T)
        
        self.prep.data_set = data_set
                
        # Assign an attribute ".data" to all new instances of Preparation
        #self.data = HomeCredit().get_data()#['train'].copy() # good practice to be sure not to modify your `data` variable
        
        #self.catcols = Preparation(self.prep.data_set).get_catcols()
        #self.numcols = Preparation(self.prep.data_set).get_numcols()
        
        self.cols = cols
        
        self.data_set = data_set
        
        """if newdf is None:
            if cols is None:
                self.data = HomeCredit().get_data()[self.prep.data_set].drop(columns= targ) #targvar does not exist            
            else:
                T = cols[:] #  [:] to keep the original list 'cols' unchanged
                T.remove(targ)
                self.data = HomeCredit().get_data()[self.prep.data_set][T] 
                         
        else:
            self.data = newdf
            print("self :", self.data.shape)
            self.cols = list(self.data.columns)
            print("ok :", self.data.shape)"""
            
        
        if newdf is None:
            if cols is None:
                self.data = HomeCredit().get_data()[self.prep.data_set] # train
                self.newdata = HomeCredit().get_data()['test'] # test
            else:
                T = cols[:] #  [:] to keep the original list 'cols' unchanged
                T.remove(targ)
                if self.prep.data_set == 'train':
                    self.data = HomeCredit().get_data()[self.data_set][cols]
                else:      
                    self.data = HomeCredit().get_data()[self.data_set][T] # we add this cond, as precaution,
                                                                       #if we went to clean  another file, and data_set
                                                                        # is not 'train'
                    print( "Please, it is better to call Cleaning(), with data_set = 'train', that includes targetVar")
                    print("Cleaning().newdata contains the data test after cleaning")
                self.newdata = HomeCredit().get_data()['test'][T] # test
                     
        else:
            if cols is None:
                self.data = HomeCredit().get_data()[self.prep.data_set]
            else:
                self.data = HomeCredit().get_data()[self.prep.data_set][cols]
            
            self.newdata = newdf
            
        
        
        #self.prep1 = Preparation(data_set, list(self.data.columns)) #targvar does not exist

        
    def get_count_missvalues(self):
        
        df = self.data#[self.prep.data_set].copy()
        
        missing_df = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
        return missing_df
    
    
    def get_percentage_missvalues(self):
        
        df = self.data#[self.prep.data_set].copy()
        
        ratio = pd.DataFrame(
            (df.isnull().sum().sort_values(ascending=False))/ df.shape[0])
        return ratio
    
    
    def plot_missvalues_table(self, na_name=False): # self : dataframe
                                            # if na_names: print the features list  
    
        df = self.data#[self.prep.data_set].copy()
        
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
    def remove_entries(self, set_df = 'train'):
        
        if set_df == 'train':
            #print("set_df____ :", set_df)
            df = self.data
        else:
            #print("set_df******:", set_df)
            df = self.newdata
        #[self.prep.data_set]#.copy()
        
        df = df[df['CODE_GENDER'] != 'XNA'] # with gender = XNA
        
        # Remove entries with DAYS_EMPLOYED > 200_000
        df = df[df['DAYS_EMPLOYED'] < 200_000]
        
        if self.cols is None:
            df = df[df['NAME_FAMILY_STATUS'] != 'Unknown'] # 'Unknown' status
            df = df[df['AMT_ANNUITY'] < 150_000]
            df = df[df['AMT_GOODS_PRICE'] < 2.5* 10**6]  
            #self.data = self.data[self.data['OWN_CAR_AGE'] < 45]
            # when removing data with CAR_AGE > 45, we got NaN value :
            # cramers_val(self, col1, col2, margins=False), col1: FLAG_OWN_CAR and col2:all columns
                   
        return df
    
    def remove_missvalues(self, set_df = 'train'):
        
        #df = self.data[self.prep.data_set].copy()
        
        #print("set_df :", set_df)
        df1 = self.remove_entries(set_df = set_df)
        #print("df1 ok :", df1.shape)
        
        self.prep1 = Preparation(self.data_set, list(df1.columns)) #targvar does not exist
            
        catcols = self.prep1.get_catcols()
        
        #print("catcols :", catcols)
        numcols = self.prep1.get_numcols()
        #print("numcols :", numcols)
        
        # Categorical Variables
        df1[catcols] = df1[catcols].replace(np.nan, '', regex=True)
        
        #print("df1 catcols ok :", list(df1.columns))
        
        #self.remove_entries()[catcols].fillna("", inplace=True)

        # Replace the NaNs in numerical column by the mean of values
        # in numerical column respectively
        df1[numcols] = df1[numcols].fillna(value=df1[numcols].mean())
        # Also, we can remove these missing values
        return df1

        
        
