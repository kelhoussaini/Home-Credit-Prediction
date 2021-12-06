import pickle
import pandas as pd

class Predict:
    
    def __init__(self, new_data):
        self.new_data = new_data

    def execute(self):
        
        df_new = self.new_data
        
        # load fitted models
        my_model = pickle.load(open("models_selected_features.pckl","rb"))   # the first one is the best is our case
        best_model = my_model[0]['Model '] # LogisticRegression(max_iter=1000) ---> fitted


        # load encoder
        my_encoder = pickle.load(open("encoder.pckl","rb"))

        catcols = [col for col in df_new.keys() if df_new[col].dtypes == "O"] # categorical features

        results = []

        for q, col_name in enumerate(catcols[:]):
               
                L = list(my_encoder[q].keys())[0]
                if L == col_name:
                    results.append((col_name, my_encoder[q][col_name]))
                    ohe = my_encoder[q][col_name]
                    new_encoded = ohe.transform(df_new[[col_name]]) # transforming data to predict
                else:
                    q = 0
                    while list(my_encoder[q].keys())[0] != col_name:
                        q = q+1
                        if list(my_encoder[q].keys())[0] == col_name:                
                            results.append((col_name, my_encoder[q][col_name]))
                            ohe = my_encoder[q][col_name]
                            new_encoded = ohe.transform(df_new[[col_name]]) # transforming data to predict


                dicts_newdata = {}
                keys = list(ohe.categories_[0])
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
                        data_res_new = data_res_new.drop(columns= col_name, axis = 1)
                        df_new = data_res_new

        # load encoder 
        my_scaler = pickle.load(open("scaler.pckl","rb"))  

        df_scaled = my_scaler.transform(df_new)

        pred = best_model.predict(df_scaled)
        
        return round(pred[0])


              




