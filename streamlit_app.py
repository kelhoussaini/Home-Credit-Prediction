import streamlit as st
import requests



import numpy as np
import pandas as pd

# this slider allows the user to select a number of lines
# to display in the dataframe

#streamlit run app.py

'''
# Home Credit Default Risk front

This front queries [Home Credit Default Risk API](http://127.0.0.1:8000/make_preds?CODE_GENDER=F&FLAG_OWN_CAR=N&OCCUPATION_TYPE=Laborers&NAME_INCOME_TYPE=Working&NAME_TYPE_SUITE=Family&EXT_SOURCE_3=0.5&DAYS_EMPLOYED=100&FLOORSMAX_AVG=0.3&DAYS_BIRTH=5000&REGION_RATING_CLIENT_W_CITY=2
)
'''

# Add a selectbox to the sidebar:
CODE_GENDER = st.sidebar.selectbox(
    'CODE_GENDER',
    ('F', 'M'))  #F      M  

FLAG_OWN_CAR = st.sidebar.selectbox(
    'FLAG_OWN_CAR',
    ('N', 'Y')) # Y      N

OCCUPATION_TYPE = st.sidebar.selectbox(
    'OCCUPATION_TYPE',
    ('Accountants', 'Cleaning staff', 'Cooking staff', 'Core staff', 'Drivers', 'HR staff', 'High skill tech staff', 'IT staff', 'Laborers', 'Low-skill Laborers', 'Managers', 'Medicine staff', 'NoValue', 'Private service staff', 'Realty agents', 'Sales staff', 'Secretaries', 'Security staff', 'Waiters/barmen staff')) #Accountants     


NAME_INCOME_TYPE = st.sidebar.selectbox(
    'NAME_INCOME_TYPE',
    ('Businessman', 'Commercial associate', 'Maternity leave', 'Pensioner', 'State servant', 'Student', 'Working'))#Working  Working

NAME_TYPE_SUITE = st.sidebar.selectbox(
    'NAME_TYPE_SUITE',
    ('Children', 'Family', 'Group of people', 'Other_A', 'Other_B', 'Spouse, partner', 'Unaccompanied'))#Unaccompanied      Group of people

   
EXT_SOURCE_3=st.number_input('EXT_SOURCE_3', value=0.445396) #0.445396         0.052036	
DAYS_EMPLOYED=st.number_input('DAYS_EMPLOYED', value=-275 ) #-275           -247
FLOORSMAX_AVG=st.number_input('FLOORSMAX_AVG', value=0.333300) # 0.333300      0.234704
DAYS_BIRTH= st.number_input('DAYS_BIRTH', value=-11163) # -11163            -16468
REGION_RATING_CLIENT_W_CITY= st.number_input('REGION_RATING_CLIENT_W_CITY', min_value=0, max_value=8, step=1, value=1)# 1    3
    
 
    
#url = 'http://127.0.0.1:8000/make_preds'
params = dict(
   CODE_GENDER=CODE_GENDER, 
        FLAG_OWN_CAR=FLAG_OWN_CAR, 
        OCCUPATION_TYPE=OCCUPATION_TYPE,
        NAME_INCOME_TYPE=NAME_INCOME_TYPE, 
        NAME_TYPE_SUITE=NAME_TYPE_SUITE, 
        EXT_SOURCE_3=EXT_SOURCE_3,
        DAYS_EMPLOYED=DAYS_EMPLOYED,
        FLOORSMAX_AVG=FLOORSMAX_AVG,
        DAYS_BIRTH=DAYS_BIRTH,
        REGION_RATING_CLIENT_W_CITY=REGION_RATING_CLIENT_W_CITY) 

# enter here the address of your initial api deployed to heroku , flask api
url = f'https://homecreditkenzael.herokuapp.com/make_preds?CODE_GENDER={CODE_GENDER}&FLAG_OWN_CAR={FLAG_OWN_CAR}&OCCUPATION_TYPE={OCCUPATION_TYPE}&NAME_INCOME_TYPE={NAME_INCOME_TYPE}&NAME_TYPE_SUITE={NAME_TYPE_SUITE}&EXT_SOURCE_3={EXT_SOURCE_3}&DAYS_EMPLOYED={DAYS_EMPLOYED}&FLOORSMAX_AVG={FLOORSMAX_AVG}&DAYS_BIRTH={DAYS_BIRTH}&REGION_RATING_CLIENT_W_CITY={REGION_RATING_CLIENT_W_CITY}'


st.write('')
st.write('')


if st.button('Predicted target'):
    response = requests.get(url, params=params)
    prediction = response.json()
    col1, col2 = st.columns(2)
    col2.metric("", f"{prediction['Prediction']}")





