import os
import sys

path_dir = (os.path.dirname(os.getcwd()))
sys.path.append(path_dir)
#print(path_dir)

import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from homecredit.predict import Predict

import pickle
import time

from fastapi import FastAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


@app.get("/make_preds")
def make_preds(CODE_GENDER, FLAG_OWN_CAR,
               OCCUPATION_TYPE, NAME_INCOME_TYPE,
               NAME_TYPE_SUITE, EXT_SOURCE_3,
               DAYS_EMPLOYED, FLOORSMAX_AVG,
               DAYS_BIRTH, REGION_RATING_CLIENT_W_CITY):

    # Create 'DF' dataframe from dictionary 'dic'
    dic = dict(
        CODE_GENDER=str(CODE_GENDER), # keys   : ['F', 'M']
        FLAG_OWN_CAR=str(FLAG_OWN_CAR), #keys   : ['N', 'Y']
        OCCUPATION_TYPE=str(OCCUPATION_TYPE), # some keys : ['Accountants', 'Cleaning staff', 'Managers', 'Medicine staff' ..
        NAME_INCOME_TYPE=str(NAME_INCOME_TYPE), # some keys   : ['Businessman', 'Commercial associate', 'Maternity leave', ...
        NAME_TYPE_SUITE=str(NAME_TYPE_SUITE), #some keys   : ['Children', 'Family', 'Group of people', ...
        EXT_SOURCE_3=float(EXT_SOURCE_3),
        DAYS_EMPLOYED=int(DAYS_EMPLOYED),
        FLOORSMAX_AVG=float(FLOORSMAX_AVG),
        DAYS_BIRTH=int(DAYS_BIRTH),
        REGION_RATING_CLIENT_W_CITY=int(REGION_RATING_CLIENT_W_CITY) )   
    
    DF = pd.DataFrame([dic]) ##Build a dataframe for the prediction
   
    #Call Predict() class that includes all the script needed for making predictions
    
    pred = Predict(new_data = DF)
        
    res = pred.execute() # with execute() method, we encode the new data based on
                          # the encoding transformation used in the train data
                          # then, we scaled the new encoded dataframe, and finally we use the model for prediction
                          # which is also used in the training, so already fitted
                
    return {"Prediction" : res}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)



#API
#http://127.0.0.1:8000/make_preds?CODE_GENDER=F&FLAG_OWN_CAR=N&OCCUPATION_TYPE=Laborers&NAME_INCOME_TYPE=Working&NAME_TYPE_SUITE=Family&EXT_SOURCE_3=0.5&DAYS_EMPLOYED=100&FLOORSMAX_AVG=0.3&DAYS_BIRTH=5000&REGION_RATING_CLIENT_W_CITY=2


# Please see API___URL-and-Features_keys.txt   to get all categorical features keys.