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
model_path = path+'models/1p-cpu-2000-3l-cgm.h5'

class LSTM_predictor(Predictor):
    name: str = "LSTM Predictor"
    pw: PredictionWindow
    prediction_values: [float]
    prediction_values_all: [float]
    model: keras.models

    def __init__(self, pw):
        super().__init__()
        self.pw: PredictionWindow = pw


       
    def calc_predictions(self, error_times: [int]) -> bool:
        

        prediction = np.array(self.pw.lstm_result)

        self.prediction_values_all = pd.Series(prediction * 500, index=range(0,181,15))
        self.prediction_values = self.prediction_values_all[error_times]
        self.prediction_values.index += 600
        self.prediction_values_all.index += 600

        # plt.plot(self.pw.data['cgmValue'])
        # plt.plot(self.prediction_values_all)
        # plt.savefig('test')
        # plt.close()
        return True
      

    def get_graph(self) -> ({'label': str, 'values': [float]}):

        return {'label': self.name, 'values': self.prediction_values_all}



