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




class Modeling:
    
    def __init__(self):
            
        path_dir = (os.path.dirname(os.getcwd()))
        sys.path.append(path_dir)
        # Assign an attribute ".data" to all new instances of Preparation
        self.data = Encoder().execute() # outliers and missing values are removed, encoding are done
        self.catcols = Preparation().get_catcols()
        self.numcols = Preparation().get_numcols()
        
    def preprocess(self): # we can here integrate scaler as arg
        
        #encoded_df = Encoder().execute()
        # create X, y
        y = self.data.TARGET
        X = self.data.drop('TARGET', axis = 1)

        # Scaling features
        scaler = MinMaxScaler() # Instanciate StandarScaler
        scaler.fit(X)

        X_rescaled = scaler.transform(X)

        # Split into Train/Test
        X_train_sc, X_test_sc, y_train, y_test = train_test_split(X_rescaled, y, test_size=0.3)
        
        return X_train_sc, X_test_sc, y_train, y_test
       
        
    def execute(self): # we can here integrate models, and scoring as arg
        
        X_train_sc, X_test_sc, y_train, y_test = self.preprocess()
        
        scoring = ['roc_auc', 'accuracy']
        
        models = []
        results = []

        # Classifiers
        models.append(('LR', LogisticRegression(max_iter=1000)))
        models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
        models.append(('DTC', DecisionTreeClassifier()))
        #models.append(('RF', RandomForestClassifier()))
        #models.append(('SVC', SVC()))

        for name, model in models:
            for s in scoring:
                model.fit(X_train_sc, y_train)
                res = cross_val_score(model, X_train_sc, y_train, cv=5, scoring=s).mean()
                results.append([name, s, res])
                print("Model: ", name, " scoring:", s, " score", res) 
                
        return {"Models": models, "Scoring":scoring, "Results": results}
    
    def predict(self, best_model, best_scoring :str):
        
        X_train_sc, X_test_sc, y_train, y_test = self.preprocess()
        
        best_model.fit(X_train_sc, y_train)
        res = cross_val_score(best_model, X_train_sc, y_train, cv=5, scoring=best_scoring).mean()
        print("score: ", res)  

        #  Predict on new data
        y_pred = best_model.predict(X_test_sc)

        # Test accuracy
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  
        
        return metrics.accuracy_score(y_test, y_pred)




