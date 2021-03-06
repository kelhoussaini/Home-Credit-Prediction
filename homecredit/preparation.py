import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from homecredit.data import HomeCredit


class Preparation:
    
    def __init__(self, data_set = 'train', cols = None):
        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)

        # Assign an attribute ".data" to all new instances of Preparation
        #self.data = HomeCredit().get_data()['train'].copy() # good practice to be sure not to modify your `data` variable
        self.data_set = data_set
        #self.cols = HomeCredit().get_data()[self.data_set].columns
        
        self.cols = cols
        
        if cols is None:
            self.data = HomeCredit().get_data()[self.data_set]
        else:
            self.data = HomeCredit().get_data()[self.data_set][cols]        
        
    def get_catcols(self):
        
        df = self.data#[self.data_set][self.cols].copy()
        
        cat_cols = [col for col in df.columns if df[col].dtypes == "O"]   
        return cat_cols
    
    def get_numcols(self):  #there are two files application_train and application_test
        
        df = self.data#[self.data_set][self.cols].copy()
        num_cols = [col for col in df.columns if (df[col].dtypes != "O")]
        
        return num_cols

    def plot_catcols_whole(self):
        
        df = self.data#[self.data_set][self.cols].copy()
        
        fig, axs = plt.subplots(len(self.get_catcols())//2, 2, figsize=(19,40)) # axs is a (1,2) nd-array
        for i, col in enumerate(self.get_catcols()[:8]):   
            sns.countplot(x = col, data = df, ax=axs[i, 0]); 

        for i, col in enumerate(self.get_catcols()[8:]):   
            sns.countplot(x = col, data = df, ax=axs[i, 1]); 

    def plot_catcols_single(self, col): # col : col name
        
        df= self.data#[self.data_set][self.cols].copy()
        
        fig, ax=plt.subplots(1,2,figsize=(14,4))

        # First plot
        sns.countplot(x = col, data = df , ax=ax[0]); 
        #ax[0].set_title(str(col) +" Entries %");

        
        # Second plot
        t = pd.crosstab(df[col], "freq", normalize=True)
        t = t.assign(type = t.index, freq = 100 * t.freq) 
        sns.barplot(y = "type", x = "freq", data = t, ax=ax[1])
        ax[1].set_title("Comparing percentages for "+str(col))
        
    def plot_numcols_single(self, col, kde=True, bins=50, nbr=1): # col : column name
        """ nbr = 365 for temporal columns"""
        
        df= self.data#[self.data_set][self.cols].copy()
        
        plt.figure(figsize = (10, 4))
        sns.histplot(df[col]/nbr, kde=kde, bins= bins);
        
    def plot_num_cat_cols(self, numcol, catcol, sample = 1000, plot_type = 1, hue = None, split=True): #target variable
        
        """ Parameters : 
        hue : "TARGET" or None"""
        
        df= self.data#[self.data_set][self.cols].copy()
        
        # it takes too long with whole data
        # to simplify visualization, we use a sample
        df_sample = df.head(sample)
        
        if plot_type == 0:
            
            fig, ax=plt.subplots(2, 1, figsize=(19, 17))
            # First plot            
            df_sample.boxplot(by=catcol,column =[numcol], 
                              grid = False, ax=ax[0]); 
            ax[0].set_title(' ')
            ax[0].set_ylabel(str(numcol));
            # Second plot
            ax = sns.boxplot(y = catcol, x = numcol, orient = "h", data = df_sample, ax=ax[1]);
            
        else:
            
            fig, ax = plt.subplots(figsize=(19, 7))
            sns.violinplot(x= catcol, y= numcol, hue = hue, split = split,  data = df_sample)
            sns.swarmplot(x= catcol, y= numcol, data=df_sample, color="White") ;
            
            
    #Proportion Table for target variable
    def targetVar(self, targ= "TARGET", other_df=None):
        #Sure, there is no target variable on test data set, but let's keep data_set='train' as arg (as precaution)
        
        # when other_df (dataframe) is true, df = other_df
        # which we need for example to plot the repartition of predicted targets 
        df= self.data#[self.data_set][self.cols].copy()
        
        if other_df is not None:
            df = other_df
        
        dataframe = (df[targ].value_counts() / len(df)).to_frame()
           
        fig, ax=plt.subplots(1,2,figsize=(14,4))

        df[targ].value_counts().plot.pie(explode=[0.1,0.1],autopct='%.2f%%',shadow=True,ax=ax[0])
        ax[0].set_title("Target Entries %")
        ax[0].set_ylabel('')

        sns.countplot(x = targ, data = df, ax=ax[1])
        ax[1].set_title('Count of Repayer Vs. defulter')

        plt.show()
        
        return dataframe
        
        
