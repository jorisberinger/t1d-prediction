import json
import logging
from datetime import timedelta
import time
import pandas
from scipy.optimize import minimize
from sklearn import metrics

import check
import extractor
import readData
import rolling
import predict
from Classes import UserData, Event
from matplotlib import pyplot as plt
import numpy as np
import cProfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

docker = False
path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data-Jens.csv"
resultPath = path + "results/"

# use example User data
udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=11, predictionlength=60,
                     stats=None)

# Set True if minimizer should get profiled
profile = False


vec_get_insulin = np.vectorize(predict.calculateBIAt, otypes=[float], excluded=[0,1,2])
vec_get_carb = np.vectorize(predict.calculateCarbAt, otypes=[float], excluded=[1,2])
def optimize():
    logger.info("start optimizing")
    # load data and select time frame
    loadData()
    logger.info("data loaded")
    directory = os.path.dirname(resultPath + "optimizer/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # set number of parameters; has to be manually done for bounds TODO Improve
    numberOfParameter = 40
    # set inital guess to 0 for all input parameters
    x0 = np.array([0] * numberOfParameter)
    # set all bounds to 0 - 1
    ub = 5
    lb = 0
    bounds = ([lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub],[lb, ub])
    logger.info(str(numberOfParameter) + " parameters set")

    # set error time points
    t_index = np.arange(0, len(cgmX_train), 3)
    t = cgmX_train[t_index]
    global real_values
    real_values = cgmY_train[t_index]
    logger.info(real_values)

    logger.info("check error at: " + str(t))
    # enable profiling
    logger.info("profiling enabled: " + str(profile))
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # get Insulin Values
    insulin_values = np.array([0] * len(t))
    t_ = t[:,np.newaxis]
    varsobject = predict.init_vars(udata.sensf, udata.idur * 60)
    for row in df.itertuples():
        iv = vec_get_insulin(row, udata, varsobject, t_).flatten()
        insulin_values = insulin_values + iv
    plt.plot(t, insulin_values)
    plt.savefig(resultPath + "optimizer/insulin.png", dpi=75)

    # Create Carb Events
    carbEvents = [];
    for i in range(0,numberOfParameter):
        carbEvents.append(Event.createCarb(i * (udata.simlength - 1) * 60 / numberOfParameter, x0[i], 60))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])

    # Minimize predicter function, with inital guess x0 and use bounds to improve speed, and constraint to positive numbers
    values = minimize(predicter, x0, args=(t_, insulin_values, ev), method='L-BFGS-B', bounds=bounds, options = {'disp': True, 'maxiter': 50})  # Set maxiter higher if you have Time
    #values = minimize(predicter, x0, method='TNC', bounds=bounds, options = {'disp': True, 'maxiter': 20})

    if profile:
        pr.disable()
        pr.dump_stats(resultPath + "optimizer/profile")
    # output values which minimize predictor
    logger.info("x_min" + str(values.x))
    # make a plot, comparing the real values with the predicter
    plot(values.x)
    # save x_min values
    with open(resultPath + "optimizer/values-1.json", "w") as file:
        file.write(json.dumps(values.x.tolist()))
        file.close()

    logger.info("finished")


def predicter(inputs, t, insulin_values, ev):
    # Calculate simulated BG for every real BG value we have. Then calculate the error and sum it up.
    # Update inputs
    logger.debug("inputs " + str(inputs))
    logger.debug("ev " + str(ev))
    ev.grams = inputs
    logger.debug("ev " + str(ev.grams))
    logger.debug(type(inputs[0]))
    logger.debug(type(ev.grams[0]))
    carb_prediction  = vec_get_carb(t, ev, udata).flatten()
    predictions = carb_prediction + insulin_values
    error = abs(real_values - predictions)
    error_sum = error.sum()
    #logger.info(real_values)

    #error_sum  = metrics.mean_squared_error(real_values, predictions)
    #error_sum = metrics.mean_absolute_error(real_values, predictions)
    logger.info("error: " + str(error_sum))
    return error_sum


def plot(values):
    logger.debug(values)
    carbEvents = []
    for i in range(0, len(values)):
        carbEvents.append(Event.createCarb(i * 15, values[i]/12, 60))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])
    # logger.info(ev)
    allEvents = pandas.concat([df, ev])
    # logger.info(allEvents)

    sim = predict.calculateBG(allEvents, udata)
    logger.debug(len(sim))
    plt.plot(sim[5], "g")
    plt.plot(cgmX, cgmY)
    logger.debug("cgmX" + str(cgmX))
    plt.savefig(resultPath + "optimizer/result-1.png", dpi=75)


def loadData():
    data = readData.read_data(filename)
    data["datetimeIndex"] = data.apply(lambda row: rolling.convertTime(row.date + ',' + row.time), axis=1)
    data["datetime"] = data["datetimeIndex"]
    data = data.set_index('datetimeIndex')
    startTime = data.index[0]
    subset = data.loc[startTime <= data.index]
    subset = subset.loc[startTime + timedelta(hours=11) > subset.index]
    global subset_train
    subset_train = subset.loc[startTime + timedelta(hours=10) > subset.index]
    global cgmX
    global cgmY
    global cgmX_train
    global cgmY_train
    cgmX , cgmY, cgmP = check.getCgmReading(subset, startTime)
    cgmX_train , cgmY_train, cgmP_train = check.getCgmReading(subset_train, startTime)
    udata.bginitial = cgmY[0]
    # Extract events
    events = extractor.getEvents(subset_train)
    converted = events.apply(lambda event: check.convertTimes(event, startTime))
    global df
    df = pandas.DataFrame([vars(e) for e in converted])

    # Remove Carb events and tempbasal
    df = df[df['etype'] != "carb"]
    df = df[df['etype'] != "tempbasal"]


if __name__ == '__main__':
    start_time = time.process_time()
    optimize()
    logger.info(str(time.process_time() - start_time) + " seconds")
