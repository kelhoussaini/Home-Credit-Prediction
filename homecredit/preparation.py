import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from homecredit.data import HomeCredit


class Preparation:
    
    def __init__(self):
        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)

        # Assign an attribute ".data" to all new instances of Preparation
        self.data = HomeCredit().get_data()['train'].copy() # good practice to be sure not to modify your `data` variable
        
    def get_catcols(self):
        
        cat_cols = [col for col in self.data.columns if self.data[col].dtypes == "O"]   
        return cat_cols
    
    def get_numcols(self):   
        num_cols = [col for col in self.data.columns if (self.data[col].dtypes != "O")]
        
        return num_cols

    def plot_catcols_whole(self):
        
        fig, axs = plt.subplots(len(self.get_catcols())//2, 2, figsize=(19,40)) # axs is a (1,2) nd-array
        for i, col in enumerate(self.get_catcols()[:8]):   
            sns.countplot(x = col, data = self.data, ax=axs[i, 0]); 

        for i, col in enumerate(self.get_catcols()[8:]):   
            sns.countplot(x = col, data = self.data, ax=axs[i, 1]); 

    def plot_catcols_single(self, col): # col : col name
        
        fig, ax=plt.subplots(1,2,figsize=(14,4))

        # First plot
        sns.countplot(x = col, data = self.data , ax=ax[0]); 
        #ax[0].set_title(str(col) +" Entries %");

        
        # Second plot
        t = pd.crosstab(self.data[col], "freq", normalize=True)
        t = t.assign(type = t.index, freq = 100 * t.freq) 
        sns.barplot(y = "type", x = "freq", data = t, ax=ax[1])
        ax[1].set_title("Comparing percentages for "+str(col))
        
    def plot_numcols_single(self, col, kde=True, bins=50): # col : col name
        
        plt.figure(figsize = (10, 4))
        sns.histplot(self.data[col], kde=kde, bins= bins);
        
    def plot_num_cat_cols(self, numcol, catcol, sample = 1000, plot_type = 1, hue = "TARGET", split=True): #target variable
        
        """ Parameters : 
        hue : "TARGET" or None"""
        
        # it takes too long with whole data
        # to simplify visualization, we use a sample
        df_sample = self.data.head(sample)
        
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
    def targetVar(self, targ= "TARGET"):
        
        dataframe = (self.data[targ].value_counts() / len(self.data)).to_frame()
           
        fig, ax=plt.subplots(1,2,figsize=(14,4))

        self.data[targ].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,ax=ax[0])
        ax[0].set_title("Target Entries %")
        ax[0].set_ylabel('')

        sns.countplot(x = targ, data = self.data, ax=ax[1])
        ax[1].set_title('Count of Repayer Vs. defulter')

        plt.show()
        
        return dataframe
        
        
