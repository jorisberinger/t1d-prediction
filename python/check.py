import logging
import os
import pandas
import numpy as np
import extractor
from datetime import datetime, timedelta

from Classes import UserData
import predict
from matplotlib import pyplot as plt

timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"

logger = logging.getLogger(__name__)

# make prediction for a window
def checkWindow(data, udata, startTime):

    # split data for only training
    trainData = data.loc[startTime + timedelta(hours=udata.simlength-1) > data.index]

    events = extractor.getEvents(trainData)
    converted = events.apply(lambda event: convertTimes(event, startTime))
    df = pandas.DataFrame([vars(e) for e in converted])
    cgmData = data[data['cgmValue'].notnull()]
    cgmData['delta'] = cgmData.apply(lambda row: getTimeDelta(row, startTime), axis=1)
    values = data[data['cgmValue'].notnull()]['cgmValue'].values
    udata.bginitial = values[0]

    bg = calculateBG(df, udata, udata.simlength * 60)


    simbg = bg[0]
    simbg_adv = bg[5]

    ## get last cgm value
    lastValue = cgmData['cgmValue'].values[len(cgmData) - 1]
    lastTime = int(cgmData['delta'].values[len(cgmData) - 1])
    logger.debug("last time: " + str(lastTime) + "\tlast value: " + str(lastValue))
    ## get second last
    secondlastValue = cgmData['cgmValue'].values[len(cgmData) - steps]
    secondlastTime = int(cgmData['delta'].values[len(cgmData) - steps])
    logger.debug("secondlast time: " + str(secondlastTime) + "\tsecondlast value: " + str(secondlastValue))
    delta = simbg[secondlastTime] - simbg[lastTime]
    delta_adv = simbg_adv[secondlastTime] - simbg_adv[lastTime]
    logger.debug("delta: " + str(delta) + "\tdelta adv: " + str(delta_adv))
    prediction = lastValue + delta
    prediction_adv = lastValue + delta_adv
    logger.debug("prediction: " + str(prediction) + "\tprediction adv: " + str(prediction_adv))
    error = lastValue - prediction
    error_adv = lastValue - prediction_adv
    logger.debug("error: " + str(error) + "\terror adv: " + str(error_adv))



    return [error, error_adv]

def getTimeDeltaFast(row, start):
    time = row.datetime
    if time < start:
        timeDifference = start - time
        return - timeDifference.seconds / 60
    else:
        timeDifference = time - start
        return timeDifference.seconds / 60
def getTimeDelta(row, start):
    if isinstance(start, datetime):
        startTime = start
    else:
        startTime = datetime.strptime(start + timeZone, timeFormat)
    if isinstance(row.datetime, datetime):
        time = row.datetime
    else:
        time = datetime.fromisoformat(row.datetime)


    if time < startTime:
        timeDifference = startTime - time
        return - timeDifference.seconds / 60
    else:
        timeDifference = time - startTime
        return timeDifference.seconds / 60


def getClosestIndex(cgmX, number):
    res = 0
    #logger.debug("cgmX last" + str(cgmX[len(cgmX) -1 ]))
    #logger.debug("number " + str(number))
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
    #logger.debug("prediction at " + str(prediction))
    return prediction


def checkFast(data, udata, startTime):

    # Get all Values of the Continuous Blood Glucose Reading, cgmX as TimeDelta from Start and cgmY the paired Value
    cgmtrue = data[data['cgmValue'].notnull()]
    cgmtrue['delta'] = cgmtrue.apply(lambda row: getTimeDeltaFast(row, startTime), axis=1)
    cgmtrue = cgmtrue[0 < cgmtrue['delta']]
    cgmtrue = cgmtrue[cgmtrue['delta'] < udata.simlength * 60]
    cgmX = cgmtrue['delta'].values
    cgmY = cgmtrue['cgmValue'].values

    # Find Data Point closest to 5h mark, save index
    if (cgmX[len(cgmX) - 1] >= (udata.simlength - 1) * 60):
        index_last_train = getClosestIndex(cgmX, (udata.simlength - 1) * 60)
        time_last_train = cgmX[index_last_train]
        train_value = cgmY[index_last_train]
    else:
        logger.warning("not able to predict")
        return None

    # Get Last Datapoint and index
    index_last_value = len(cgmY) - 1  # TODO check if index l v - index l t is circa 60 min
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
    prediction_last_train = np.array(predict.calculateBGAt(time_last_train, df_train, udata, udata.simlength * 60))
    logger.debug("prediction train " + str(prediction_last_train))
    # Get prediction Value for last value
    prediction_last_value = np.array(predict.calculateBGAt(time_last_value, df_train, udata, udata.simlength * 60))
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

def checkCurrent(data, udata, startTime):

    run_plot = False

    events = extractor.getEvents(data)

    converted = events.apply(lambda event: convertTimes(event, startTime))
    df = pandas.DataFrame([vars(e) for e in converted])
    df = df[df.time > 0]
    df = df[df.time < udata.simlength * 60]
    df_train = df[df.time < (udata.simlength - 1) * 60]

    cgmtrue = data[data['cgmValue'].notnull()]
    cgmtrue['delta'] = cgmtrue.apply(lambda row: getTimeDelta(row, startTime), axis=1)
    cgmtrue = cgmtrue[0 < cgmtrue['delta']]
    cgmtrue = cgmtrue[cgmtrue['delta'] < udata.simlength * 60]
    cgmX = cgmtrue['delta'].values
    cgmY = cgmtrue['cgmValue'].values

    initialBG = cgmY[0]

    udata.bginitial = initialBG

    data = predict.calculateBG(df_train, udata, udata.simlength * 60)

    basalValues = df[df.etype == 'tempbasal']
    carbValues = df[df.etype == 'carb']
    bolusValues = df[df.etype == 'bolus']


    #for i in range(0,len(data[0])):
     #   print(str(data[3][i]) + '\t' * 5 + str(data[0][i]) + '\t' * 5 + str(data[1][i]) + '\t' * 5 + str(data[2][i]))

    dataasdf = pandas.DataFrame([data[1], data[0], data[1], data[2]])

    if(cgmX[len(cgmX) -1] >= (udata.simlength -1) * 60):
        index = getClosestIndex(cgmX, (udata.simlength -1) * 60)
    else:
        logger.warning("not able to predict")
        return None

    prediction_vals = getPredictionVals(cgmX, cgmY, index,  data[0])
    prediction_vals_adv = getPredictionVals(cgmX, cgmY, index,  data[5])

    if run_plot:
        plt.figure(figsize=(10, 7))
        plt.grid(color="#cfd8dc")
        plt.xlim(0,udata.simlength * 60)

        plt.plot(cgmX, cgmY, "#263238", alpha=0.8, label="real BG")

        plt.plot(data[3], data[0], "#b71c1c", alpha=0.5, label="sim BG")
        plt.plot(range(int(cgmX[index]), len(data[3])), prediction_vals, "#b71c1c", alpha=0.8, label="SIM BG Pred")
        plt.plot(data[3], data[5], "#4527a0", alpha=0.5, label="sim BG ADV")
        plt.plot(range(int(cgmX[index]), len(data[3])), prediction_vals_adv, "#4527a0", alpha=0.8, label="SIM BG Pred ADV")

        # vertical prediction
        plt.axhline(y = prediction_vals[0], xmin=5/6,alpha=0.8, label="Same Value Prediction")


        plt.plot(data[3], data[1], "#64dd17", alpha=0.5, label="sim BGC")
        plt.plot(data[3], data[2], "#d50000", alpha=0.5, label="sim BGI")
        plt.plot(data[3], data[4], "#aa00ff", alpha=0.5, label="sim BGI ADV")




        plt.plot(basalValues.time, [0] * len(basalValues), "bo", alpha=0.8, label="basal event (not used)")
        plt.plot(carbValues.time, [0] * len(carbValues), "go", alpha=0.8, label="carb event")
        plt.plot(bolusValues.time, [0] * len(bolusValues), "ro", alpha=0.8, label="bolus evnet")



        plt.legend(loc=2, bbox_to_anchor=(1, 1))
        plt.tight_layout(pad=6)
        plt.axvline(x=(udata.simlength - 1) * 60, color="black")
        directory = os.path.dirname("/t1d/results/")
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig("/t1d/results/result-"+startTime.strftime('%Y-%m-%d-%H-%M')+".svgz", dpi=300)

        plt.close()

    lastValue = cgmY[len(cgmY)-1]
    prediction = prediction_vals[int(cgmX[len(cgmX)-1] - cgmX[index])]
    prediction_adv = prediction_vals_adv[int(cgmX[len(cgmX) - 1] - cgmX[index])]
    error = lastValue - prediction
    error_adv = lastValue - prediction_adv
    error_same_value = lastValue - cgmY[index]
    logger.debug("error: " + str(error) +  "\terror_adv: " + str(error_adv) + "\terror_same_value: " + str(error_same_value))
    if error is None:
        logger.warning("none type error")
    if error_adv is None:
        logger.warning("none type adv")
    if error_same_value is None:
        logger.warning("none type sv")
    return [error, error_adv, error_same_value]


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
        event.time = event.time - 30
    if event.etype == "tempbasal":
        t1 = datetime.strptime(event.t1 + timeZone, timeFormat)
        event.t1 = event.time
        #timeDifference = eventTime - t1
        #event.t1 = timeDifference.seconds
        #t2 = datetime.strptime(event.t2, timeFormat)
        #timeDifference = eventTime - t2
        event.t2 = event.t1 + 30 # TODO better estimate
    #print(event.time, event.t1, event.t2)
    return event