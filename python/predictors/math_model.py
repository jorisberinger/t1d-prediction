import logging

import PredictionWindow


class MathPredictor:
    prediction_values: [float]
    name: str = "Math Model Predictor"
    def __init__(self, pw: PredictionWindow):
        logging.info("init optimizer")
        self.pw: PredictionWindow = pw

    def calc_predictions(self, error_times: [int]) -> bool:
        logging.info("get Errors at {}".format(error_times))

    def get_errors_and_curve(self, error_times: [int]) -> bool:
        logging.info("get Errors at {} and curve".format(error_times))

