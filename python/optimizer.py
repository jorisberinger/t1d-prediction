import json
import logging
from datetime import timedelta
import time
import pandas
from scipy.optimize import minimize
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
filename = path + "data/csv/data.csv"
resultPath = path + "results/"

# use example User data
udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=11, predictionlength=60,
                     stats=None)

# Set True if minimizer should get profiled
profile = False

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
    bounds = ([0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1])
    logger.info(str(numberOfParameter) + " parameters set")
    # enable profiling
    logger.info("profiling enabled: " + str(profile))
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # Minimize predicter function, with inital guess x0 and use bounds to improve speed, and constraint to positive numbers
    values = minimize(predicter, x0, method='L-BFGS-B', bounds=bounds, options = {'disp': True, 'maxiter': 30})  # Set maxiter higher if you have Time
    #values = minimize(predicter, x0, method='TNC', bounds=bounds, options = {'disp': True, 'maxiter': 20})

    if profile:
        pr.disable()
        pr.print_stats()
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

def predicter(inputs):
    logger.debug(inputs)
    carbEvents = []
    # Create Carb event every 15 Minutes and combine it with the insulin events
    for i in range(0,len(inputs)):
        carbEvents.append(Event.createCarb(i* udata.simlength * 60  / len(inputs), inputs[i], 60))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])
    allEvents = pandas.concat([df,ev])
    logger.debug("All Events " + str(allEvents))

    # Total Error
    error = 0
    # Calculate simulated BG for every real BG value we have. Then calculate the error and sum it up.
    # TODO change number of points to increase speed. Speed vs Acc Tradeoff
    for i in range(0,len(cgmX)):  # TODO try to use itertuple() to iterate #faster
        # calculate simulated value
        simValue = predict.calculateBGAt(int(cgmX[i]), allEvents, udata)[1]  # TODO compare [0] and [1]
        logger.debug("sim value " + str(simValue))
        # get real value
        realValue = cgmY[i]
        logger.debug("real value " + str(realValue))
        error += abs(realValue - simValue)  # TODO try out different error functions

    logger.info("error: " + str(error))
    return error

def plot(values):
    logger.debug(values)
    carbEvents = []
    for i in range(0, len(values)):
        carbEvents.append(Event.createCarb(i * 15, values[i], 60))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])
    # logger.info(ev)
    allEvents = pandas.concat([df, ev])
    # logger.info(allEvents)

    sim = predict.calculateBG(allEvents, udata)
    logger.debug(len(sim))
    plt.plot(sim[5], "g")S
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
    cgmX , cgmY, cgmP = check.getCgmReading(subset, startTime)
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
