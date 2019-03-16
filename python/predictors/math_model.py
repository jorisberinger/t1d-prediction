import logging

import PredictionWindow
from predictors.predictor import Predictor


class MathPredictor(Predictor):
    prediction_values: [float]
    name: str = "Math Model Predictor"

    def __init__(self, pw: PredictionWindow):
        super().__init__()
        logging.info("init optimizer")
        self.pw: PredictionWindow = pw

    def calc_predictions(self, error_times: [int]) -> bool:
        logging.info("get Errors at {}".format(error_times))

    def get_graph(self) -> ({'label': str, 'values': [float]}):
        logging.info("get graph")

