import PredictionWindow
from predictors.predictor import Predictor
import pandas as pd


# DB3P
# PATIENT TEST MEAN : 111.7890048358361
# PATIENT TRAIN MEAN : 129.18711973769572

# DB1P
# PATIENT MEAN : 155.9000831574234


class Mean_predictor(Predictor):
    pw: PredictionWindow
    prediction_values: [float]
    name: str = "Mean Value Predictor"
    mean_value: float

    def __init__(self, pw: PredictionWindow):
        super().__init__()
        self.pw = pw
        self.mean_value = 111.7890048358361

    def calc_predictions(self, error_times: [int]) -> bool:
        self.prediction_values = pd.Series([self.mean_value] * len(error_times),
                                           index=error_times + self.pw.userData.train_length())
        return True
    def get_graph(self) -> ({'label': str, 'values': [float]}):
        res = self.prediction_values.append(pd.Series(self.mean_value, index = [self.pw.userData.train_length()]))
        return {'label': self.name, 'values': res}



