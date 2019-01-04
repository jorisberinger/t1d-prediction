import pandas
import logging
from datetime import datetime, timedelta
from autotune_prep import convertTime
from check import checkWindow, checkCurrent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

filename = "../../Data/data1217.csv"
def getDateTime(row):
    return convertTime(row.date + ',' + row.time)


def predictWindow(window):
    print(window.shape)
    return 1


def predictRolling2(inputData, userData):
    # user rolling window to get 5 hours of data
    logger.debug("convert data to dataFrame")
    data = pandas.DataFrame(inputData)
    logger.debug("data frame ")
    data["datetime"] = data.apply(lambda row: getDateTime(row), axis=1)
    data = data.set_index('datetime')
    data["datetime"] = data.apply(lambda row: getDateTime(row), axis=1)
    # make sample size smaller
    data = data.loc[data.index > datetime(year=2017,month=12,day=15)]

    # user rolling window
    rolling = data.rolling('5h')
    rolling.apply(lambda window: predictWindow(window))

# make rolling prediction and call checkWindow for every data window
def rolling(data, delta, udata):
    startTime = data.index[0]
    startTime = startTime.replace(minute= startTime.minute // int((udata.predictionlength)))
    endTime = data.index[len(data)-1]
    results = []

    # loop through the data
    while startTime < endTime - timedelta(hours=udata.simlength):

        # select data for this window
        subset = data.loc[startTime <= data.index]
        subset = subset.loc[startTime + timedelta(hours=udata.simlength) > subset.index]

        # call the prediction method
        results.append(checkCurrent(subset, udata, startTime))

        startTime += delta # delta determines the time between two predictions
    return results


# prepare data for rolling prediction and call rolling prediction
def predictRolling(inputData, userData):
        # user rolling window to get 5 hours of data
        logger.debug("convert data to dataFrame")
        data = pandas.DataFrame(inputData)
        logger.debug("data frame ")
        data["datetime"] = data.apply(lambda row: getDateTime(row), axis=1)
        data = data.set_index('datetime')
        data["datetime"] = data.apply(lambda row: getDateTime(row), axis=1)
        #logger.debug(data)
        # make sample size smaller
        logger.debug(len(data))
        # user rolling window
        return rolling(data, timedelta(minutes=15), userData)


