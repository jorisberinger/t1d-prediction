import pandas
import logging
from datetime import datetime, timedelta
from autotune_prep import convertTime
from check import checkWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

filename = "../../Data/data1217.csv"
def getDateTime(row):
    return convertTime(row.date + ',' + row.time)


def predictWindow(window):
    print(window.shape)
    return 1


def predictRolling(inputData, userData):
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


def rolling(data, delta, steps, udata, stepsback):
    startTime = data.index[0]
    startTime = startTime.replace(minute= startTime.minute // int((steps.seconds / 60)))
    endTime = data.index[len(data)-1]
    results = []
    while startTime < endTime - delta:
        subset = data.loc[startTime <= data.index]
        subset = subset.loc[startTime + delta > subset.index]

        results.append(checkWindow(subset, udata, startTime, stepsback))

        startTime += steps
    return results
def predictRollingCached(inputData, userData, steps):
        # user rolling window to get 5 hours of data
        logger.debug("convert data to dataFrame")
        data = pandas.DataFrame(inputData)
        logger.debug("data frame ")
        data["datetime"] = data.apply(lambda row: getDateTime(row), axis=1)
        data = data.set_index('datetime')
        data["datetime"] = data.apply(lambda row: getDateTime(row), axis=1)
        logger.debug(data)
        # make sample size smaller
        # user rolling window
        return rolling(data, timedelta(hours=5), timedelta(minutes=15), userData, steps)


