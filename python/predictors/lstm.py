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
model_path = path+'models/test-100-3l-cgm, insulin, carbs, optimized, tod.h5'

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
        features.index = list(map(float , features.index))
        features = features.sort_index()
        features['time_of_day'] = get_time_of_day(self.pw.startTime)
        features['features-90'] = 0
        for i, v in enumerate(range(0,600,15)):
            features.loc[v,'features-90'] = data_object['features-90'][i]

        x = np.empty((1,features.shape[0], features.shape[1]))
        x[0] = features
        prediction = self.model.predict(x)
        self.prediction_values_all = pd.Series(prediction[0] * 500, index=range(0,len(prediction[0])*5,5))
        self.prediction_values = self.prediction_values_all[error_times]
        self.prediction_values.index += 600
        self.prediction_values_all.index += 600
        return True
      

    def get_graph(self) -> ({'label': str, 'values': [float]}):

        return {'label': self.name, 'values': self.prediction_values_all}


def get_feature_list(data_object):
    df = data_object['data'][['cgmValue', 'basalValue', 'bolusValue', 'mealValue']].fillna(0)
    df.index = list(map(float , df.index))
    df = df.sort_index()
    df['features-90'] = 0
    df['time_of_day'] = get_time_of_day(data_object['start_time'])
    for i, v in enumerate(range(0,600,15)):
        df.loc[v,'features-90'] = data_object['features-90'][i]

    return df