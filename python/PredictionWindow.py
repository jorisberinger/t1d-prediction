import logging

import pandas
import numpy as np
import extractor


class PredictionWindow:
    data: pandas.DataFrame = None
    data_long: pandas.DataFrame = None
    startTime = None
    endTime = None
    userData = None
    plot = None
    cgmX = None
    cgmY = None
    index_last_train = None
    time_last_train = None
    train_value = None
    index_last_value = None
    time_last_value = None
    lastValue = None
    events = None
    prediction = None
    errors = None
    real_values: [float] = None
    features_90: [float]
    lstm_result: [float]
    def set_values(self, error_times: np.array):
        # Get all Values of the Continuous Blood Glucose Reading, cgmX as TimeDelta from Start and cgmY the paired Value
        self.cgmY = get_cgm_reading(self.data)

        # Set initial Blood Glucose Level
        self.userData.bginitial = self.cgmY[0]

        # Get Train Datapoint
        self.train_value = self.data.at[float(self.userData.train_length()), 'cgmValue']
        # Get last Datapoint
        self.lastValue = self.data.at[float(self.userData.simlength * 60), 'cgmValue']

        # Get Events for Prediction
        self.events = extractor.getEventsAsDataFrame(self)

        # Get values at Error Times
        self.real_values = self.cgmY[error_times + self.userData.train_length()]

def get_cgm_reading(data):
        cgm_true = data[data['cgmValue'].notnull()]
        if len(cgm_true) > 0:
            cgm_y = cgm_true['cgmValue']
            return cgm_y
        else:
            return None
