FROM python:3.8.6-buster

COPY api /api
COPY Home-Credit-Prediction /Home-Credit-Prediction

COPY homecredit/predict.py /homecredit/predict.py


COPY api/models_selected_features.pckl /models_selected_features.pckl
COPY api/encoder.pckl /encoder.pckl
COPY api/scaler.pckl /scaler.pckl

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT


#EXPOSE 8080
#CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8080"]




