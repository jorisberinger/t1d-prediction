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
        self.iob = iob
        self.cob = cob
        return {'label': self.name, 'values': values[5]}

    def optimize_mix(self) -> (int, pandas.DataFrame, pandas.DataFrame):
        # Set time steps where to calculate the error between real_values and prediction
        # Every 15 minutes including start and end
        t_error = get_error_time_steps(self.pw, 5)

        # Get Real Values
        real_values = get_real_values(self.pw, t_error)

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
        values = minimize(predictor, x0, args = (real_values, insulin_values, patient_carb_matrix), method = 'L-BFGS-B',
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
        self.prediction_values = list(map(lambda error_time: self.get_prediction_value(error_time, allEvents), error_times + self.pw.userData.train_length()))
        self.all_events = allEvents

    def optimize_only(self):
        # Optimize without prediction
        self.optimize_mix()
        carb_events = get_carb_events(self.carb_values, self.carb_types, self.t_carb_events)
        insulin_events = self.pw.events[self.pw.events.etype != 'carb']
        allEvents = pandas.concat([insulin_events, carb_events])
        self.all_events = allEvents
        return self.carb_values


def predictor(inputs, real_values, insulin_values, p_cob):
    # Calculate simulated BG for every real BG value we have. Then calculate the error and sum it up.
    # Update inputs
    carb_values = np.array(np.matmul(inputs, p_cob))
    predictions = carb_values + insulin_values
    error = np.absolute(real_values - predictions.flatten())
    error_sum = np.matmul(error, error)
    #error_sum = error.sum()
    return error_sum
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

def optimize(pw: PredictionWindow, carb_duration: int) -> int:
    # set error time points
    t = np.arange(0, pw.userData.simlength * 60 - pw.userData.predictionlength, 15)
    # logger.debug(t)

    real_values = np.array(pw.cgmY.loc[t])
    # logger.info("real values " + str(real_values))

    # set number of parameters
    numberOfParameter = 40
    # set inital guess to 0 for all input parameters
    x0 = np.array([1] * numberOfParameter)
    # set all bounds to 0 - 1
    ub = 20
    lb = 0
    bounds = np.array([(lb, ub)] * numberOfParameter)
    # logger.debug("bounds " + str(bounds))
    # logger.info(str(numberOfParameter) + " parameters set")

    # logger.info("check error at: " + str(t))
    # enable profiling
    # logger.info("profiling enabled: " + str(profile))
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # get Insulin Values
    insulin_events = pw.events[pw.events.etype == 'bolus']
    insulin_values = np.array([pw.cgmY[0]] * len(t))
    t_ = t[:, np.newaxis]
    varsobject = predict.init_vars(pw.userData.sensf, pw.userData.idur * 60)
    for row in insulin_events.itertuples():
        iv = vec_get_insulin(row, udata, varsobject, t_).flatten()
        insulin_values = insulin_values + iv

    # create Time Matrix
    times = []
    for i in t:
        times.append(predict.vec_cob1(t - i, carb_duration))
    cob_matrix = np.matrix(times)
    # logger.debug(cob_matrix)
    # logger.info(t.shape)

    # get patient coefficient
    patient_coefficient = udata.sensf / udata.cratio

    # multiply patient_coefficient with cob_matrix
    patient_carb_matrix = patient_coefficient * cob_matrix
    # logger.debug(patient_carb_matrix)

    # Minimize predicter function, with inital guess x0 and use bounds to improve speed, and constraint to positive numbers
    values = minimize(predictor, x0, args = (real_values, insulin_values, patient_carb_matrix), method = 'L-BFGS-B',
                      bounds = bounds,
                      options = {'disp': False, 'maxiter': 1000})  # Set maxiter higher if you have Time
    # values = minimize(predicter, x0, args=(t_, insulin_values, patient_carb_matrix), method='TNC', bounds=bounds, options = {'disp': True, 'maxiter': 1000})

    if profile:
        pr.disable()
        pr.dump_stats(resultPath + "optimizer/profile")
    # output values which minimize predictor
    # logger.info("x_min" + str(values.x))
    # make a plot, comparing the real values with the predicter
    # plot(values.x, t)
    # save x_min values
    # with open(resultPath + "optimizer/values-1.json", "w") as file:
    #    file.write(json.dumps(values.x.tolist()))
    #    file.close()

    prediction_value = getPredictionValue(values.x, t, pw, carb_duration)
    # logger.info("prediction value: " + str(prediction_value))
    # logger.info("finished")
    # plot(values.x, t, pw)
    carb_events = pandas.Series(values.x, index = t)
    if not pw.plot:
        return prediction_value
    else:
        prediction_curve = getPredictionCurve(values.x, t, pw, carb_duration)
        return prediction_value, prediction_curve, carb_events





def getPredictionCurve(carb_values: [float], t: [float], predictionWindow: PredictionWindow, carb_durations: [int]) -> ([float], pandas.DataFrame):
    carb_events = get_carb_events(carb_values, carb_durations, t)
    # remove original carb events from data
    insulin_events = predictionWindow.events[predictionWindow.events.etype != 'carb']
    allEvents = pandas.concat([insulin_events, carb_events])

    values, iob, cob = predict.calculateBG(allEvents, predictionWindow.userData)
    return values[5], carb_events






def get_carb_events(carb_values, carb_durations, t) -> pandas.DataFrame:
    carbEvents = []
    number_event_times = int(len(carb_values) / len(carb_durations))
    for i in range(0, number_event_times):
        for index, carb_duration in enumerate(carb_durations):
            carbEvents.append(Event.createCarb(t[i], carb_values[i + number_event_times * index] / 12, carb_duration))
    carb_events = pandas.DataFrame([vars(e) for e in carbEvents])
    return carb_events


def optimizeMain():
    logger.info("start optimizing")
    # load data and select time frame
    loadData()
    logger.info("data loaded")
    directory = os.path.dirname(resultPath + "optimizer/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # set error time points
    t_index = np.arange(0, len(cgmX_train), 3)
    t = cgmX_train[t_index]
    #global real_values
    real_values = cgmY_train[t_index]
    logger.info(real_values)

    # set number of parameters
    numberOfParameter = len(t)
    # set inital guess to 0 for all input parameters
    x0 = np.array([1] * numberOfParameter)
    # set all bounds to 0 - 1
    ub = 20
    lb = 0
    bounds = np.array([(lb, ub)] * numberOfParameter)
    logger.debug("bounds " + str(bounds))
    logger.info(str(numberOfParameter) + " parameters set")

    logger.info("check error at: " + str(t))
    # enable profiling
    logger.info("profiling enabled: " + str(profile))
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # get Insulin Values
    insulin_values = np.array([cgmY[0]] * len(t))
    t_ = t[:, np.newaxis]
    varsobject = predict.init_vars(udata.sensf, udata.idur * 60)
    for row in df.itertuples():
        iv = vec_get_insulin(row, udata, varsobject, t_).flatten()
        insulin_values = insulin_values + iv
    plt.plot(t, insulin_values)
    plt.savefig(resultPath + "optimizer/insulin.png", dpi = 75)
    plt.close()

    # Create Carb Events
    carbEvents = [];
    for i in range(0, numberOfParameter):
        carbEvents.append(Event.createCarb(i * (udata.simlength - 1) * 60 / numberOfParameter, x0[i], carb_duration))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])

    # create Time Matrix
    times = []
    for i in t:
        times.append(predict.vec_cob1(t - i, carb_duration))
    cob_matrix = np.matrix(times)
    logger.debug(cob_matrix)
    logger.info(t.shape)

    # get patient coefficient
    p_co = udata.sensf / udata.cratio

    # multiply p_co with cob_matrix
    p_cob = p_co * cob_matrix
    logger.debug(p_cob)

    # Minimize predicter function, with inital guess x0 and use bounds to improve speed, and constraint to positive numbers
    values = minimize(predicter, x0, args = (real_values, insulin_values, p_cob), method = 'L-BFGS-B', bounds = bounds,
                      options = {'disp': True, 'maxiter': 1000})  # Set maxiter higher if you have Time
    # values = minimize(predicter, x0, args=(t_, insulin_values, p_cob), method='TNC', bounds=bounds, options = {'disp': True, 'maxiter': 1000})

    if profile:
        pr.disable()
        pr.dump_stats(resultPath + "optimizer/profile")
    # output values which minimize predictor
    logger.info("x_min" + str(values.x))
    # make a plot, comparing the real values with the predicter
    plot(values.x, t)
    # save x_min values
    with open(resultPath + "optimizer/values-1.json", "w") as file:
        file.write(json.dumps(values.x.tolist()))
        file.close()

    logger.info("finished")


def plot(values, t, pw: PredictionWindow):
    logger.debug(values)
    carbEvents = []
    for i in range(0, len(values)):
        carbEvents.append(Event.createCarb(t[i], values[i] / 12, carb_duration))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])
    # logger.info(ev)
    insulin_events = pw.events[pw.events.etype != 'carb']
    original_carbs = pw.events[pw.events.etype == 'carb']
    allEvents = pandas.concat([insulin_events, ev])
    # logger.info(allEvents)

    sim = predict.calculateBG(allEvents, udata)
    logger.debug(len(sim))

    # Plot
    basalValues = allEvents[allEvents.etype == 'tempbasal']
    carbValues = allEvents[allEvents.etype == 'carb']
    bolusValues = allEvents[allEvents.etype == 'bolus']

    fig = plt.figure(figsize = (12, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios = [3, 1, 1])
    # fig, ax = plt.subplots()

    ax = plt.subplot(gs[0])
    plt.xlim(0, udata.simlength * 60 + 1)
    plt.ylim(0, 400)
    plt.grid(color = "#cfd8dc")
    # Major ticks every 20, minor ticks every 5
    major_ticks_x = np.arange(0, udata.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, udata.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, 401, 50)
    # minor_ticks_x = np.arange(0, 400, 15)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor = True)
    ax.set_yticks(major_ticks_y)

    plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False)
    plt.box(False)

    # And a corresponding grid
    # ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which = 'minor', alpha = 0.2)
    ax.grid(which = 'major', alpha = 0.5)

    # Plot Line when prediction starts
    plt.axvline(x = (udata.simlength - 1) * 60, color = "black")
    # Plot real blood glucose readings
    plt.plot(pw.cgmY, "#263238", alpha = 0.8, label = "real BG")
    # Plot sim results
    plt.plot(sim[5], "g", alpha = 0.8, label = "sim BG")

    # Plot Legend
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.tight_layout(pad = 6)

    ax = plt.subplot(gs[1])

    major_ticks_x = np.arange(0, udata.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, udata.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, 21, 4)
    # minor_ticks_x = np.arange(0, 400, 15)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor = True)
    ax.set_yticks(major_ticks_y)

    ax.grid(which = 'minor', alpha = 0.2)
    ax.grid(which = 'major', alpha = 0.5)

    plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False)
    plt.box(False)

    # Plot Events
    plt.xlim(0, udata.simlength * 60 + 1)
    plt.ylim(0, 20)
    plt.grid(color = "#cfd8dc")
    if (len(basalValues) > 0):
        plt.bar(basalValues.time, basalValues.dbdt, 5, alpha = 0.8, label = "basal event (not used)")
    if len(carbValues) > 0:
        plt.bar(carbValues.time, carbValues.grams, 5, alpha = 0.8, label = "carb event")
    if len(bolusValues) > 0:
        plt.bar(bolusValues.time, bolusValues.units, 5, alpha = 0.8, label = "bolus event")
    plt.bar(original_carbs.time, original_carbs.grams, 5, alpha = 0.8, label = "original Carb")

    # plt.bar(basalValues.time, [0] * len(basalValues), "bo", alpha=0.8, label="basal event (not used)")
    # plt.plot(carbValues.time, [0] * len(carbValues), "go", alpha=0.8, label="carb event")
    # plt.plot(bolusValues.time, [0] * len(bolusValues), "ro", alpha=0.8, label="bolus evnet")
    # Plot Line when prediction starts
    plt.axvline(x = (udata.simlength - 1) * 60, color = "black")
    # Plot Legend
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.tight_layout(pad = 6)
    plt.subplots_adjust(hspace = 0.2)

    ax = plt.subplot(gs[2])

    major_ticks_x = np.arange(0, udata.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, udata.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, 51, 10)
    # minor_ticks_x = np.arange(0, 400, 15)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor = True)
    ax.set_yticks(major_ticks_y)

    ax.grid(which = 'minor', alpha = 0.2)
    ax.grid(which = 'major', alpha = 0.5)

    plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False)
    plt.box(False)

    # Plot Events
    plt.xlim(0, udata.simlength * 60 + 1)
    plt.ylim(0, 50)
    plt.grid(color = "#cfd8dc")

    # plot error values
    # err = error.tolist()[0]
    # plt.bar(t, err, 5, color="#CC0000", alpha=0.8, label="error")
    plt.axvline(x = (udata.simlength - 1) * 60, color = "black")
    # Plot Legend
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.tight_layout(pad = 6)
    plt.subplots_adjust(hspace = 0.2)

    plt.savefig(resultPath + "optimizer/result-" + str(pw.startTime) + ".png", dpi = 150)


def loadData():
    data = readData.read_data(filename)
    data["datetimeIndex"] = data.apply(lambda row: rolling.convertTime(row.date + ',' + row.time), axis = 1)
    data["datetime"] = data["datetimeIndex"]
    data = data.set_index('datetimeIndex')
    startTime = data.index[0]
    subset = data.loc[startTime <= data.index]
    subset = subset.loc[startTime + timedelta(hours = 11) > subset.index]
    global subset_train
    subset_train = subset.loc[startTime + timedelta(hours = 10) > subset.index]
    global cgmX
    global cgmY
    global cgmX_train
    global cgmY_train
    cgmX, cgmY = check.getCgmReading(subset)
    cgmX_train, cgmY_train, cgmP_train = check.getCgmReading(subset_train, startTime)
    udata.bginitial = cgmY[0]
    # Extract events
    events = extractor.getEvents(subset_train)
    converted = events.apply(lambda event: check.convertTimes(event, startTime))
    global df
    df = pandas.DataFrame([vars(e) for e in converted])

    # Remove Carb events and tempbasal
    global original_carbs
    original_carbs = df[df['etype'] == 'carb']
    df = df[df['etype'] != "carb"]
    df = df[df['etype'] != "tempbasal"]


if __name__ == '__main__':
    start_time = time.process_time()
    optimizeMain()
    logger.info(str(time.process_time() - start_time) + " seconds")


