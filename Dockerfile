FROM python:3.8.6-buster
#FROM tiangolo/uvicorn-gunicorn:python3.7

#WORKDIR /streamlit
#use the above line for streamlit.. branch

# Copy local code to the container image.
#ENV APP_HOME /app


COPY api /api
COPY Home-Credit-Prediction /Home-Credit-Prediction

COPY homecredit/predict.py /homecredit/predict.py


COPY api/models_selected_features.pckl /models_selected_features.pckl
COPY api/encoder.pckl /encoder.pckl
COPY api/scaler.pckl /scaler.pckl

#RUN mkdir /streamlit 
#use the above line for streamlit.. branch

COPY streamlit_app.py /streamlit_app.py
#use the above line for streamlit.. branch

COPY requirements.txt /requirements.txt


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# --------------- Configure Streamlit ---------------
#RUN mkdir -p /root/.streamlit


#RUN bash -c 'echo kenza'
#
EXPOSE 8501
#EXPOSE 8080

#COPY . /streamlit
#use the above lines for streamlit.. branch

#EXPOSE 8501
#use the above lines for streamlit.. branch

#CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
CMD streamlit run streamlit_app.py 


#EXPOSE 8080
#CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8080"]
