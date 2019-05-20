import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMAResults

from PredictionWindow import PredictionWindow
from predictors.predictor import Predictor

import keras
path = os.getenv('T1DPATH', '../')
logger = logging.getLogger(__name__)
model_path = path+'model-2000.h5'

class LSTM(Predictor):
    name: str = "LSTM Predictor"
    pw: PredictionWindow
    prediction_values: [float]
    prediction_values_all: [float]
    model: keras.models

    def __init__(self, pw):
        super().__init__()
        self.pw: PredictionWindow = pw
        self.model = keras.models.load_model(model_path)
       
    def calc_predictions(self, error_times: [int]) -> bool:
        data = self.pw.data.iloc[:600:5]
        features =  data[['cgmValue', 'basalValue', 'bolusValue', 'mealValue']].fillna(0)
        features['cgmValue'] /= 500
        x = np.empty((1,features.shape[0], features.shape[1]))
        x[0] = features
        prediction = self.model.predict(x)
        self.prediction_values_all = pd.Series(prediction[0] * 500, index=range(0,len(prediction[0])*5,5))
        self.prediction_values = self.prediction_values_all[error_times]
        self.prediction_values.index += 600
        return True
      

    def get_graph(self) -> ({'label': str, 'values': [float]}):

        return {'label': self.name, 'values': self.prediction_values_all}

