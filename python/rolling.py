import pandas
import logging
from datetime import timedelta

from Classes import PredictionWindow
from autotune.autotune_prep import convertTime
import check

logger = logging.getLogger(__name__)


# make rolling prediction and call checkWindow for every data window
def rolling(data, delta, udata, autotune_res, plotOption):
    predictionWindow = PredictionWindow()
    predictionWindow.data = data

    # select starting point as first data point, always start at 15 minute intervals
    startTime = data.index[0]
    startTime = startTime.replace(minute= data.index[0].minute // int(udata.predictionlength))

    endTime = data.index[len(data)-1]
    predictionWindow.endTime = data.index[len(data)-1]
    results = []
    i = 0
    # loop through the data
    while startTime < endTime - timedelta(hours=udata.simlength):
        logger.info("#" + str(i))
        i += 1
        predictionWindow.startTime = startTime
        predictionWindow.endTime = startTime + timedelta(hours=udata.simlength)
        # select data for this window
        subset = data.loc[startTime <= data.index]
        subset = subset.loc[predictionWindow.endTime > subset.index]
        predictionWindow.data = subset

        # Set Sensitivity factor and CarbRatio
        arStart = autotune_res[subset.date.values[0]]
        arEnd = autotune_res[subset.date.values[len(subset.date.values) -1]] # TODO use second day if overlapping

        udata.cratio = arStart['cr']
        udata.sensf = arStart['sens'][0]['sensitivity']
        udata.basalProfile = arStart['basal']
        logger.debug("date " + subset.date.values[0] + "\cratio " + str(udata.cratio) + "\tsensf " + str(udata.sensf))
        predictionWindow.userData = udata
        predictionWindow.plot = plotOption
        # call the prediction method
        res = check.checkAndPlot(predictionWindow)
        if res is not None:
            results.append(res)
        startTime += delta  # delta determines the time between two predictions
    logger.debug("length of result " + str(len(results)))
    return results


# prepare data for rolling prediction and call rolling prediction
def predictRolling(inputData, userData, autotune_res, plotOption):
        # user rolling window to get 5 hours of data
        # convert data to dataFrame
        data = pandas.DataFrame(inputData)

        # Create new index with datetime
        data["datetimeIndex"] = data.apply(lambda row: convertTime(row.date + ',' + row.time), axis=1)
        data["datetime"] = data["datetimeIndex"]
        data = data.set_index('datetimeIndex')

        # Run rolling window prediction with 15 minute intervals
        res = rolling(data, timedelta(minutes=15), userData, autotune_res,  plotOption)
        # user rolling window
        return res


