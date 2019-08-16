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

class Error_predictor(Predictor):
    name: str = "Error Predictor"
    pw: PredictionWindow
    prediction_values: [float]
    prediction_values_all: [float]

    def __init__(self, pw):
        super().__init__()
        self.pw: PredictionWindow = pw

       
    def calc_predictions(self, error_times: [int]) -> bool:
        

        prediction_error = np.array(self.pw.error_result)


        self.prediction_values_all = self.pw.real_values + prediction_error
        self.prediction_values = self.pw.real_values + prediction_error

        return True
      

    def get_graph(self) -> ({'label': str, 'values': [float]}):

        return {'label': self.name, 'values': self.prediction_values_all}
