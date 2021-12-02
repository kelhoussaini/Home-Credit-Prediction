import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from homecredit.data import HomeCredit
from homecredit.preparation import Preparation
from homecredit.cleaner import Cleaning
from homecredit.exploration import Exploration
from homecredit.encoder import Encoder

import pickle
import time

import shutil

class Modeling:
    
    def __init__(self, data_set = 'train', cols = None, newdf = None, targ= "TARGET"): # if newdf is None we get the two data sets in the 
                                                                        # directory : test.csv and train.csv
        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)
        # Assign an attribute ".data" to all new instances of Preparation

         # Preparation
        #self.prep = Preparation(data_set, cols)    
        
        self.cols = cols
        
        # Cleaning
        #self.cl = Cleaning(data_set, cols)

        #self.cl.prep.data_set = data_set
        
        #print("self.cl.prep.data_set    ")
        
        self.en = Encoder(data_set, cols, newdf, targ= targ)#(data_set =data_set, cols = cols, newdf = newdf)
                        
        #print("self.en calling    ", self.en.data.shape, self.en.new_data.shape)
        #(self.data, self.new_data) = self.en.execute(data_topredict=True)
        self.data = self.en.execute(data_topredict=True)[0]
        self.new_data = self.en.execute(data_topredict=True)[1]

        #print("self.en execute    ")
  
        
    def preprocess(self, VarTarg = 'TARGET', scaler = MinMaxScaler(), data_topredict=False): # we can here integrate scaler as arg
        
        #encoded_df = Encoder().execute()
        # create X, y
        y = self.data[VarTarg]
        X = self.data.drop(VarTarg, axis = 1)
        
        
        # Split into Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        

        # Scaling features
        #scaler = MinMaxScaler() # Instanciate StandarScaler
        scaler.fit(X_train)

        X_train_sc = scaler.transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        if data_topredict:
            #encoded_df_pred  = self.en.execute(data_topredict=True)[1]
            encoded_df_pred_sc = scaler.transform(self.new_data)

        res = (X_train_sc, X_test_sc, y_train, y_test)
        return (res ,encoded_df_pred_sc) if data_topredict else res
       
        
    def execute(self, models=[], scoring = ['accuracy']): # we can here integrate models, and scoring as arg
        #scoring = ['roc_auc', 'accuracy']
        X_train_sc, X_test_sc, y_train, y_test = self.preprocess()
        
        #scoring = ['roc_auc', 'accuracy']
        
        #models = []
        results = []

        # Classifiers
        #models.append(('LR', LogisticRegression(max_iter=1000)))
        #models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
        #models.append(('DTC', DecisionTreeClassifier()))
        
        #models.append(('RF', RandomForestClassifier()))
        #models.append(('SVC', SVC()))

        filename  = ["all_features" if self.cols is None else "selected_features"][0]
        
        with open("models_"+filename+".pckl", "wb") as f:
            for name, model in models:
                for s in scoring:
                    # start timer
                    start = time.time()

                    model.fit(X_train_sc, y_train)
                    train_res = cross_val_score(model, X_train_sc, y_train, cv=5, scoring=s).mean()
                    
                    # stop timing
                    end = time.time()
                    time_run = (end - start)/60
    
                    results.append({"Name ": name, "Model " : model, 
                                    " scoring": s, " train score" : train_res,
                                     "time_run (mins)": time_run} )
                    print("Model: ", name, " scoring:", s, " train score", train_res, "time_run (mins)", time_run) 
                    
            # Save the models
            #dictResults = {"Models": models, "Scoring":scoring, "Results": results}
            
            pickle.dump(results, f) 
        
        #print(f.name)
        #pathh = os.path.dirname(os.getcwd())
        #print(pathh)
        #target = f.name
                    
        #shutil.copyfile(f.name, target)

        
         
        return results
    
    def evaluate_model(self, best_model=LogisticRegression(max_iter=1000), best_scoring = 'accuracy'): # #scoring = ['roc_auc', 'accuracy']
        
        X_train_sc, X_test_sc, y_train, y_test = self.preprocess()
        
        best_model.fit(X_train_sc, y_train)

        #  Predict on test data
        y_pred = best_model.predict(X_test_sc)

        # model accuracy
        if best_scoring == 'roc_auc':
            print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  
            testScore = metrics.roc_auc_score(y_test, y_pred)
        
        testScore = metrics.accuracy_score(y_test, y_pred)
        
        return testScore
                
    
    def predict_newdata(self, best_model):
        
        ((X_train_sc, X_test_sc, y_train, y_test), encoded_df_pred_sc) = self.preprocess(data_topredict=True)
        
        #= self.preprocess(data_topredict=True)[1]
        
        best_model.fit(X_train_sc, y_train)
        
        #  Predict on test data
        y_pred = best_model.predict(encoded_df_pred_sc)

        return {"Predictions" : y_pred}
    





