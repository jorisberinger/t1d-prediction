import cProfile
import json
import logging
import os
import time
from datetime import timedelta

import numpy as np
import pandas
from matplotlib import pyplot as plt, gridspec
from scipy.optimize import minimize

import check
import extractor
from predictors import predict
import rolling
from Classes import UserData, Event
from PredictionWindow import PredictionWindow
from data import readData
from predictors.predictor import Predictor



logger = logging.getLogger(__name__)

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data.csv"
resultPath = path + "results/"

# use example User data
udata = UserData(bginitial = 100.0, cratio = 5, idur = 4, inputeeffect = None, sensf = 41, simlength = 11,
                 predictionlength = 60,
                 stats = None)

# Set True if minimizer should get profiled
profile = False

# get vectorized forms of function for predicter
vec_get_insulin = np.vectorize(predict.calculateBIAt, otypes = [float], excluded = [0, 1, 2])
vec_get_carb = np.vectorize(predict.calculateCarbAt, otypes = [float], excluded = [1, 2])


class Optimizer(Predictor):
    carb_values: [float]
    pw:PredictionWindow
    prediction_values: [float]
    t_carb_events: [int]
    all_events: pandas.DataFrame
    name: str = "Optimizer Mixed Carb Types Predictor"
    iob: []
    cob: []

    def __init__(self, pw: PredictionWindow, carb_types: [int]):
        super().__init__()
        logger.debug("init optimizer")
        self.pw: PredictionWindow = pw
        self.carb_types = carb_types
        self.name = self.name + ' ' + str(carb_types)

    def calc_predictions(self, error_times: [int]) -> bool:
        logger.debug("get Errors at {}".format(error_times))
        self.optimize_mix()
        self.get_prediction_values(error_times)
        logger.debug(self.prediction_values)
        return True

    def get_graph(self) -> ({'label': str, 'values': [float]}, {'label': str, 'events': [float]}):
        logger.debug("get graph")
        values, iob, cob = predict.calculateBG(self.all_events, self.pw.userData)
        vals = values[5]
        self.iob = iob
        self.cob = cob
        offset = vals[self.pw.userData.train_length()] - self.pw.train_value
        vals[self.pw.userData.train_length():] = vals[self.pw.userData.train_length():] - offset
        return {'label': self.name, 'values': vals}

    def optimize_mix(self) -> (int, pandas.DataFrame, pandas.DataFrame):
        # Set time steps where to calculate the error between real_values and prediction
        # Every 15 minutes including start and end
        t_error = get_error_time_steps(self.pw, 5)

        # Get Real Values
        real_values = get_real_values(self.pw, t_error)
        
        # set weights
        weights = np.array(list(map(lambda x: (x + 1) / len(real_values)/2+0.5,range(len(real_values)))))


        # Number of Parameters per Carb Type
        t_carb_events = get_carb_time_steps(self.pw, 15)
        parameter_count = len(t_carb_events) * len(self.carb_types)

        # Array to hold input variables, one for every carb event at every time step and for every carb type
        # set initial guess to 0 for all input parameters
        x0 = np.array([0] * parameter_count)

        # Define lower and upper bound for variables
        lb, ub = 0, 20
        bounds = np.array([(lb, ub)] * parameter_count)

        # Get Insulin Values
        insulin_events, insulin_values = get_insulin_events(self.pw, t_error)

        # Create carbs on Board Matrix
        cob_matrix = get_cob_matrix(t_carb_events, t_error, self.carb_types)

        # multiply patient_coefficient with cob_matrix
        patient_carb_matrix = udata.sensf / udata.cratio * cob_matrix

        # Minimize predicter function, with inital guess x0 and use bounds to improve speed, and constraint to positive numbers
        values = minimize(predictor, x0, args = (real_values, insulin_values, patient_carb_matrix, weights), method = 'L-BFGS-B',
                          bounds = bounds, options = {'disp': False, 'maxiter': 1000, 'maxfun': 1000000, 'maxls': 200})

        logger.debug("success {}".format(values.success))
        self.carb_values =  values.x
        self.t_carb_events = t_carb_events

    def get_prediction_value(self, error_time: int, all_events: pandas.DataFrame) -> float:
        value = predict.calculateBGAt0(error_time, all_events, self.pw.userData)
        return value[1]

    def get_prediction_values(self, error_times: np.array):
        carb_events = get_carb_events(self.carb_values, self.carb_types, self.t_carb_events)
        insulin_events = self.pw.events[self.pw.events.etype != 'carb']
        allEvents = pandas.concat([insulin_events, carb_events])
        offset = self.get_prediction_value(self.pw.userData.train_length(), allEvents) - self.pw.train_value
        self.prediction_values = list(map(lambda error_time: self.get_prediction_value(error_time, allEvents) - offset, error_times + self.pw.userData.train_length()))

        self.all_events = allEvents

    def optimize_only(self):
        # Optimize without prediction
        self.optimize_mix()
        carb_events = get_carb_events(self.carb_values, self.carb_types, self.t_carb_events)
        insulin_events = self.pw.events[self.pw.events.etype != 'carb']
        allEvents = pandas.concat([insulin_events, carb_events])
        self.all_events = allEvents
        return self.carb_values


def predictor(inputs, real_values, insulin_values, p_cob, weights):
    # Calculate simulated BG for every real BG value we have. Then calculate the error and sum it up.
    # Update inputs
    carb_values = np.array(np.matmul(inputs, p_cob))
    predictions = carb_values + insulin_values
    error = np.absolute(real_values - predictions.flatten())
    squared_error = np.power(error, 2)
    weighted_error = np.matmul(squared_error, weights)
    # error_sum = np.matmul(error, error)
    #error_sum = error.sum()
    return weighted_error
# return time steps with step size step_size in the range from sim_length training period
def get_error_time_steps(pw: PredictionWindow, step_size: int) -> np.array:
    t = np.arange(0, pw.userData.simlength * 60 - pw.userData.predictionlength + 1, step_size)
    return t

# return time steps with step size step_size in the range from sim_length training period
def get_carb_time_steps(pw: PredictionWindow, step_size: int) -> np.array:
    t = np.arange(0, pw.userData.simlength * 60 - pw.userData.predictionlength, step_size)
    return t


# returns the real cgm values at the times steps given
def get_real_values(pw: PredictionWindow, time_steps: np.array) -> np.array:
    return np.array(pw.cgmY.loc[time_steps])


# calculates insulin values and extracts insulin events include initial value
def get_insulin_events(pw: PredictionWindow, t: np.array) -> (np.array, np.array):
    # extract all bolus events from all events
    insulin_events = pw.events[pw.events.etype == 'bolus']
    # set insulin values to initial blood glucose level
    insulin_values = np.array([pw.cgmY[0]] * len(t))

    t_ = t[:, np.newaxis]
    varsobject = predict.init_vars(pw.userData.sensf, pw.userData.idur * 60)
    for row in insulin_events.itertuples():
        iv = vec_get_insulin(row, udata, varsobject, t_).flatten()
        insulin_values = insulin_values + iv

    return insulin_events, insulin_values


# creates matrix with Carb on Board value for every carb event at every error check timestep
def get_cob_matrix(t_carb: np.array, t_error: np.array, carb_durations: [int]) -> np.matrix:
    cob_values = []
    for carb_duration in carb_durations:
        for i in t_carb:
            cob_values.append(predict.vec_cob1(t_error - i, carb_duration))

    return np.matrix(cob_values)



def get_carb_events(carb_values, carb_durations, t) -> pandas.DataFrame:
    carbEvents = []
    number_event_times = int(len(carb_values) / len(carb_durations))
    for i in range(0, number_event_times):
        for index, carb_duration in enumerate(carb_durations):
            carbEvents.append(Event.createCarb(t[i], carb_values[i + number_event_times * index] / 12, carb_duration))
    carb_events = pandas.DataFrame([vars(e) for e in carbEvents])
    return carb_events



