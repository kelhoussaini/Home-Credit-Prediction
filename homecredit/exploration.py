import os
import sys
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from homecredit.data import HomeCredit
from homecredit.cleaner import Cleaning
from homecredit.preparation import Preparation

from scipy.stats import chi2_contingency # need this for chi-squared function



class Exploration:
       
    def __init__(self):
         # Assign an attribute ".data" to all new instances of Preparation
        #self.data = HomeCredit().get_data()['train'].copy() # good practice to be sure not to modify your `data` variable
        
        # Cleaning
        self.data = Cleaning().remove_entries()
        self.catcols = Preparation().get_catcols()
       
        
    def plot_correlation(self):
    
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(self.data.corr(), cmap='coolwarm', annot = False, label = 'small', cbar = False)
        ax.set_title('Correlation Matrix')
        plt.show() ;
        
    def confusion_matrix(self, col1: str, col2: str, annot=False, margins=True):
        
        chisqt = pd.crosstab(self.data[col1], self.data[col2], margins=margins)
        
        plt.figure(figsize = (9, 7))
        sns.heatmap(chisqt, annot = annot,  cmap = "BuPu");
    
        return chisqt
    
    def categ_relation(self, col1:str, col2:str, margins=False):
        #col1, col2 :Categorical Variables
        chisqt = pd.crosstab(self.data[col1], self.data[col2], margins=margins)
        value = chisqt.to_numpy() 
        
        ## NULL hypothesis : variables are independent of each other.
        stat, p, dof= chi2_contingency(value)[0:3]

        #print("degree of freedom", dof)

        alpha = 0.05
        #print("p value: " + str(p)) 
        if p <= alpha: 
            print('Reject NULL HYPOTHESIS : Variables are dependent of each other') 
        else: 
            print('ACCEPT NULL HYPOTHESIS : Variables are independent of each other')

        return stat, p, dof
    
    def cramers_val(self, col1, col2, margins=False): # df : dataframe
        chisqt = pd.crosstab(self.data[col1], self.data[col2], margins=margins)
        value = chisqt.to_numpy() 

        #Chi-squared test statistic, sample size, and minimum of rows and columns
        X2 = chi2_contingency(value, correction=False)[0]
        n = np.sum(value)
        minDim = min(value.shape)-1

        #calculate Cramer's V 
        V = np.sqrt((X2/n) / minDim) 

        return V
    
    # Plot of Heatmap of Cramer's V
    def plot_heatmapCramerV(self):
        L = len(self.catcols)
        cramers_outputs = np.zeros((L,L))

        for i,x in enumerate(self.catcols):
            for j,y in enumerate(self.catcols):
                result = round(self.cramers_val(x,y,margins=False),4)
                cramers_outputs[i,j] = result
                
        fig = plt.figure(figsize = (8, 8))  # instanciate figure for heat map
        ax = sns.heatmap(cramers_outputs, annot = True,  cmap = "BuPu", fmt=".0%", cbar = False)
        ax.set_xticklabels(self.catcols)
        ax.set_yticklabels(self.catcols)
        ax.tick_params(axis = 'x', labelrotation = 90)
        ax.tick_params(axis = 'y', labelrotation = 0)
        ax.set_title("Heatmap of Cramer's V on categorical variables");