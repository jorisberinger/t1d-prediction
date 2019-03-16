import PredictionWindow
from predictors.predictor import Predictor
import pandas as pd

class SameValue(Predictor):
    pw: PredictionWindow
    prediction_values: [float]
    name: str = "Same Value Predictor"

    def __init__(self, pw: PredictionWindow):
        super().__init__()
        self.pw = pw

    def calc_predictions(self, error_times: [int]) -> bool:
        self.prediction_values = pd.Series([self.pw.train_value] * len(error_times),
                                           index=error_times + self.pw.userData.train_length())

    def get_graph(self) -> ({'label': str, 'values': [float]}):
        res = self.prediction_values.append(pd.Series(self.pw.train_value, index = [self.pw.userData.train_length()]))
        return {'label': self.name, 'values': res}
