import logging
import os
import pandas
import numpy as np
import extractor
from datetime import datetime
import predict
from matplotlib import pyplot as plt
from matplotlib import gridspec


logger = logging.getLogger(__name__)

timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"


def getTimeDelta(row, start):
    time = row.datetime
    if time < start:
        timeDifference = start - time
        return - timeDifference.seconds / 60
    else:
        timeDifference = time - start
        return timeDifference.seconds / 60


def getClosestIndex(cgmX, number):
    res = 0
    while cgmX[res] <= number:
        res += 1
    return res - 1


def getPredictionVals(cgmX, cgmY, index,  simData):
    zero = simData[int(cgmX[index])]
    start = cgmY[index]
    res = []
    for i in range(int(cgmX[index]), len(simData)):
        res.append(start + simData[i] - zero)
    return res


def getPredictionAt(index_last_train, df_train, udata):
    prediction = predict.calculateBGAt(index_last_train, df_train, udata, udata.simlength * 60)
    return prediction


def convertTimes(event, start):
    if isinstance(start, datetime):
        startTime = start
    else:
        startTime = datetime.strptime(start + timeZone, timeFormat)

    eventTime = datetime.strptime(event.time + timeZone, timeFormat)
    timeDifference = eventTime - startTime
    if eventTime < startTime:
        timeDifference = startTime - eventTime
        event.time = - timeDifference.seconds / 60
    else:
        event.time = timeDifference.seconds / 60

    if event.etype == "carb":
        event.time = event.time - 30  # TODO verify
    if event.etype == "tempbasal":
        event.t1 = event.time
        event.t2 = eventTime
    return event

def getCgmReading(data, startTime):
    cgmtrue = data[data['cgmValue'].notnull()]
    logger.debug("Length: " + str(len(cgmtrue)))
    if len(cgmtrue > 0):
        cgmtrue['delta'] = cgmtrue.apply(lambda row: getTimeDelta(row, startTime), axis=1)
        cgmX = cgmtrue['delta'].values
        cgmY = cgmtrue['cgmValue'].values
        return cgmX, cgmY
    else:
        return None, None

def checkFast(data, udata, startTime):

    # Get all Values of the Continuous Blood Glucose Reading, cgmX as TimeDelta from Start and cgmY the paired Value
    cgmX, cgmY = getCgmReading(data, startTime)
    # Check if there is data
    if cgmX is None:
        return None
    # Check if there is data around the 5h mark
    if not (cgmX[len(cgmX) - 1] >= (udata.simlength - 1) * 60):
        logger.warning("not able to predict")
        return None

    # Set initial Blood Glucose Level
    udata.bginitial = cgmY[0]

    # Get Train Datapoint
    index_last_train = getClosestIndex(cgmX, (udata.simlength - 1) * 60)
    time_last_train = cgmX[index_last_train]
    train_value = cgmY[index_last_train]

    # Get last Datapoint
    index_last_value = len(cgmY) - 1  # TODO check if index l v - index l t is appr. 60 min
    time_last_value = cgmX[index_last_value]
    lastValue = cgmY[index_last_value]

    # Get Events for Prediction
    events = extractor.getEvents(data)
    converted = events.apply(lambda event: convertTimes(event, startTime))
    df = pandas.DataFrame([vars(e) for e in converted])
    df_train = df[df.time < (udata.simlength - 1) * 60]

    logger.debug("index train " + str(index_last_train))
    logger.debug("index last " + str(index_last_value))
    # Get prediction Value for last train value
    prediction_last_train = np.array(predict.calculateBGAt(time_last_train, df_train, udata))
    logger.debug("prediction train " + str(prediction_last_train))
    # Get prediction Value for last value
    prediction_last_value = np.array(predict.calculateBGAt(time_last_value, df_train, udata))
    logger.debug("prediction value " + str(prediction_last_value))
    # Get Delta between train and last value
    prediction_delta = prediction_last_value - prediction_last_train
    logger.debug("delta " + str(prediction_delta))
    # add on last Train value
    prediction = np.add(train_value, prediction_delta)
    logger.debug("prediction " + str(prediction))
    # add same value prediction
    prediction = np.append(prediction, train_value)
    logger.debug("prediction " + str(prediction))
    # calculate error
    errors = np.subtract(lastValue, prediction)
    logger.debug("errors " + str(errors))

    return errors.tolist()

def checkAndPlot(data, udata, startTime):
    # Get all Values of the Continuous Blood Glucose Reading, cgmX as TimeDelta from Start and cgmY the paired Value
    cgmX, cgmY = getCgmReading(data, startTime)

    # Check if there is data
    if cgmX is None:
        return None
    # Check if there is data around the 5h mark
    if not (cgmX[len(cgmX) - 1] >= (udata.simlength - 1) * 60):
        logger.warning("not able to predict")
        return None

    # Set initial Blood Glucose Level
    udata.bginitial = cgmY[0]

    # Get Train Datapoint
    index_last_train = getClosestIndex(cgmX, (udata.simlength - 1) * 60)
    time_last_train = int(cgmX[index_last_train])
    train_value = cgmY[index_last_train]

    # Get last Datapoint
    index_last_value = len(cgmY) - 1  # TODO check if index l v - index l t is appr. 60 min
    time_last_value = int(cgmX[index_last_value])
    lastValue = cgmY[index_last_value]

    # Get Events for Prediction
    events = extractor.getEvents(data)
    converted = events.apply(lambda event: convertTimes(event, startTime))
    df = pandas.DataFrame([vars(e) for e in converted])
    df_train = df[df.time < (udata.simlength - 1) * 60]

    # Run Prediction
    data = predict.calculateBG(df_train, udata)

    basalValues = df[df.etype == 'tempbasal']
    carbValues = df[df.etype == 'carb']
    bolusValues = df[df.etype == 'bolus']

    # Get prediction Value for last train value
    prediction_last_train = np.array([data[0][time_last_train], data[5][time_last_train]])
    logger.debug("prediction train " + str(prediction_last_train))
    # Get prediction Value for last value
    prediction_last_value = np.array([data[0][time_last_value], data[5][time_last_value]])
    logger.debug("prediction value " + str(prediction_last_value))
    # Get Delta between train and last value
    prediction_delta = prediction_last_value - prediction_last_train
    logger.debug("delta " + str(prediction_delta))
    # add on last Train value
    prediction = np.add(train_value, prediction_delta)
    logger.debug("prediction " + str(prediction))
    # add same value prediction
    prediction = np.append(prediction, train_value)
    logger.debug("prediction " + str(prediction))
    # calculate error
    errors = np.subtract(lastValue, prediction)
    logger.debug("errors " + str(errors))

    # get values for prediction timeframe
    prediction_vals = getPredictionVals(cgmX, cgmY, index_last_train, data[0])
    prediction_vals_adv = getPredictionVals(cgmX, cgmY, index_last_train, data[5])

    # Plot

    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    #fig, ax = plt.subplots()

    ax = plt.subplot(gs[0])
    plt.xlim(0, udata.simlength * 60)
    plt.ylim(0, 400)
    plt.grid(color="#cfd8dc")
    # Major ticks every 20, minor ticks every 5
    major_ticks_x = np.arange(0, udata.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, udata.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, 401, 50)
    #minor_ticks_x = np.arange(0, 400, 15)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
    plt.box(False)

    # And a corresponding grid
    #ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


    # Plot Line when prediction starts
    plt.axvline(x=(udata.simlength - 1) * 60, color="black")
    # Plot real blood glucose readings
    plt.plot(cgmX, cgmY, "#263238", alpha=0.8, label="real BG")
    # Plot sim results
    plt.plot(data[3], data[0], "#b71c1c", alpha=0.5, label="sim BG")
    plt.plot(range(int(cgmX[index_last_train]), len(data[3])), prediction_vals, "#b71c1c", alpha=0.8, label="SIM BG Pred")
    plt.plot(data[3], data[5], "#4527a0", alpha=0.5, label="sim BG ADV")
    plt.plot(range(int(cgmX[index_last_train]), len(data[3])), prediction_vals_adv, "#4527a0", alpha=0.8, label="SIM BG Pred ADV")
    # Same value prediction
    plt.axhline(y=prediction_vals[0], xmin=5/6, alpha=0.8, label="Same Value Prediction")

    # Plot Legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=6)


    # Plot Insulin and Carb graph
    #plt.plot(data[3], data[1], "#64dd17", alpha=0.5, label="sim BGC")
    #plt.plot(data[3], data[2], "#d50000", alpha=0.5, label="sim BGI")
    #plt.plot(data[3], data[4], "#aa00ff", alpha=0.5, label="sim BGI ADV")

    ax = plt.subplot(gs[1])

    major_ticks_x = np.arange(0, udata.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, udata.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, 11, 2)
    # minor_ticks_x = np.arange(0, 400, 15)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
    plt.box(False)

    # Plot Events
    plt.xlim(0, udata.simlength * 60)
    plt.ylim(0, 10)
    plt.grid(color="#cfd8dc")
    logger.debug(basalValues.values[0])
    if(len(basalValues) > 0):
        plt.bar(basalValues.time, basalValues.dbdt, 5, alpha=0.8, label="basal event (not used)")
    logger.debug(carbValues)
    if len(carbValues) > 0:
        plt.bar(carbValues.time, carbValues.grams, 5, alpha=0.8, label="carb event")
    if len(bolusValues) > 0:
        plt.bar(bolusValues.time, bolusValues.units, 5, alpha=0.8, label="bolus event")
    #plt.bar(basalValues.time, [0] * len(basalValues), "bo", alpha=0.8, label="basal event (not used)")
    #plt.plot(carbValues.time, [0] * len(carbValues), "go", alpha=0.8, label="carb event")
    #plt.plot(bolusValues.time, [0] * len(bolusValues), "ro", alpha=0.8, label="bolus evnet")
    # Plot Line when prediction starts
    plt.axvline(x=(udata.simlength - 1) * 60, color="black")
    # Plot Legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=6)
    plt.subplots_adjust(hspace=0.2)
    # Save plot as svgz (smallest format, able to open with chrome)
    directory = os.path.dirname("/t1d/results/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig("/t1d/results/plots/result-"+startTime.strftime('%Y-%m-%d-%H-%M')+".png", dpi=75)
    plt.close()

    return errors.tolist()
