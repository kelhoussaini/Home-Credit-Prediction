from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy
app = Flask(__name__)

@app.route('/prediction_api', methods=["GET"])
def prediction_api():
    try:
        # Getting the paramters from API call
        Votes_value = float(request.args.get('Votes'))
        Average_Cost_value=float(request.args.get('Average_Cost'))
        Price_range_value=float(request.args.get('Price_range'))
                
        # Calling the funtion to get predictions
        prediction_from_api=FunctionGeneratePrediction(
                                                     inp_Votes=Votes_value,
                                                     inp_Average_Cost=Average_Cost_value,
                                                     inp_Price_range=Price_range_value
                                                )

        return (prediction_from_api)
    
    except Exception as e:
        return('Something is not right!:'+str(e))
