from PredictionWindow import PredictionWindow
from predictors.predictor import Predictor
import pandas as pd
import numpy as np


class LastNDelta(Predictor):
    pw: PredictionWindow
    prediction_values: [float]
    name: str = "Last {} Predictor"
    time_step: int
    polynomial: np.polynomial

    def __init__(self, pw: PredictionWindow, time_step: int = 30):
        super().__init__()
        self.pw = pw
        self.name = self.name.format(time_step)
        self.time_step = time_step

    def calc_predictions(self, error_times: [int]) -> bool:
        last_n_time = self.pw.userData.train_length() - self.time_step
        last_n_value = self.pw.cgmY[last_n_time]

        last_time = self.pw.userData.train_length()
        last_value = self.pw.train_value

        coefficients = np.polyfit([last_n_time, last_time], [last_n_value, last_value],1)

        polynomial = np.poly1d(coefficients)

        prediction_values = polynomial(error_times + self.pw.userData.train_length())
        self.prediction_values = pd.Series(prediction_values,
                                           index=error_times + self.pw.userData.train_length())
        self.polynomial = polynomial
        return True
    
    def get_graph(self) -> ({'label': str, 'values': [float]}):

        res = self.prediction_values.append(pd.Series(self.pw.train_value, index = [self.pw.userData.train_length()]))
        return {'label': self.name, 'values': res}
