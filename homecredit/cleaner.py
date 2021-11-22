import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from homecredit.data import HomeCredit
from homecredit.preparation import Preparation

    
    
class Cleaning:
    
    def __init__(self):
        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)
        # Assign an attribute ".data" to all new instances of Preparation
        self.data = HomeCredit().get_data()['train'].copy() # good practice to be sure not to modify your `data` variable
        self.catcols = Preparation().get_catcols()
        self.numcols = Preparation().get_numcols()
        
    def get_count_missvalues(self):
        missing_df = pd.DataFrame(self.data.isnull().sum().sort_values(ascending=False))
        return missing_df
    
    
    def get_percentage_missvalues(self):
        ratio = pd.DataFrame(
            (self.data.isnull().sum().sort_values(ascending=False))/ self.data.shape[0])
        return ratio
    
    
    def plot_missvalues_table(self, na_name=False): # self : dataframe
                                            # if na_names: print the features list  
    
        na_cols = [col for col in self.data.columns if self.data[col].isnull().sum() > 0]

        count = self.data[na_cols].isnull().sum().sort_values(ascending=False)
        ratio = ( self.data[na_cols].isnull().sum() / self.data.shape[0] * 100 ).sort_values(ascending=False)
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
        self.data = self.data[self.data['CODE_GENDER'] != 'XNA'] # with gender = XNA
        self.data = self.data[self.data['NAME_FAMILY_STATUS'] != 'Unknown'] # 'Unknown' status
        # Remove entries with DAYS_EMPLOYED > 200_000
        self.data = self.data[self.data['DAYS_EMPLOYED'] < 200_000]
        self.data = self.data[self.data['AMT_ANNUITY'] < 150_000]
        self.data = self.data[self.data['AMT_GOODS_PRICE'] < 2.5* 10**6]  
        self.data = self.data[self.data['OWN_CAR_AGE'] < 45]
        return self.data
    
    def remove_missvalues(self):
        
        df = self.remove_entries()
        # Categorical Variables
        df[self.catcols].fillna('', inplace=True)

        # Replace the NaNs in numerical column by the mean of values
        # in numerical column respectively
        df[self.numcols] = self.data[self.numcols].fillna(value=df[self.numcols].mean())
        return df

        
        
