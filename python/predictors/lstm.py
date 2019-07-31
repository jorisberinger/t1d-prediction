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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import CuDNNLSTM, Dropout, Dense, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import keras
path = os.getenv('T1DPATH', '../')
logger = logging.getLogger(__name__)
model_path = path+'models/2p-cpu-2000-3l-cgm, insulin, carbs, optimized, tod.h5'

class LSTM_predictor(Predictor):
    name: str = "LSTM Predictor"
    pw: PredictionWindow
    prediction_values: [float]
    prediction_values_all: [float]
    model: keras.models

    def __init__(self, pw):
        super().__init__()
        self.pw: PredictionWindow = pw
        self.model = keras.models.load_model(model_path)
        #self.model = get_lstm_model(6)
        #self.model.load_weights(model_path)

       
    def calc_predictions(self, error_times: [int]) -> bool:
        data = self.pw.data.iloc[:600:5]
        features =  data[['cgmValue', 'basalValue', 'bolusValue', 'mealValue']].fillna(0)
        features['cgmValue'] /= 500
        features.index = list(map(float , features.index))
        features = features.sort_index()
        features['features-90'] = 0
        features['time_of_day'] = get_time_of_day(self.pw.startTime)
        
        for i, v in enumerate(range(0,600,15)):
            features.loc[v,'features-90'] = self.pw.features_90[i]

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




def get_lstm_model(number_features):
    

    model = Sequential()
    model.add(LSTM(units = 120, input_shape = (120, number_features), return_sequences= True, recurrent_activation='sigmoid')) 
    model.add(Dropout(0.5))
    model.add(LSTM(units = 40, return_sequences=True, recurrent_activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(LSTM(40, recurrent_activation='sigmoid'))
    model.add(Dense(37))
    model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy', 'mae'])
    model.summary()
    return model

def get_time_of_day(start_time:str)->pd.Series:
    logging.debug("get Time of day")
    time = pd.Timestamp(start_time)
    time_of_day = pd.Series([0] * 780)
    for offset in range(780):
        logging.debug("offset: {}".format(offset))
        t = time + pd.Timedelta('{}M'.format(offset))
        logging.debug("time: {}".format(t))
        hour = t.hour
        logging.debug("hour: {}".format(hour))
        category = int((hour - 2) / 4)
        logging.debug("cat: {}".format(category))
        time_of_day[offset] = category

    return time_of_day