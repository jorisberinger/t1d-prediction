import pandas as pd
import numpy as np
import logging
from datetime import timedelta

from Classes import PredictionWindow, UserData
from autotune.autotune_prep import convertTime
import check
from data import checkData

logger = logging.getLogger(__name__)


# make rolling prediction and call checkWindow for every data window
def rolling(data: pd.DataFrame, delta: pd.Timedelta, user_data: UserData, autotune_res: dict, plotOption: bool):

    predictionWindow = PredictionWindow()

    # select starting point as first data point, always start at 15 minute intervals
    startTime = data.index[0]
    startTime = startTime.replace(minute=data.index[0].minute)

    endTime = data.index[len(data)-1]
    predictionWindow.endTime = data.index[len(data)-1]
    results = []
    i = 0
    # loop through the data
    while startTime < endTime - timedelta(hours=user_data.simlength):
        logger.info("#" + str(i))
        i += 1
        predictionWindow.startTime = startTime
        predictionWindow.endTime = startTime + timedelta(hours=user_data.simlength)
        # select data for this window
        subset = data.loc[startTime <= data.index]
        subset = subset.loc[predictionWindow.endTime >= subset.index]


        # Set Sensitivity factor and CarbRatio
        if len(subset) > 0:

            arStart = autotune_res[subset.date.values[0]]
            #arEnd = autotune_res[subset.date.values[len(subset.date.values) -1]]  # TODO use second day if overlapping
            user_data.cratio = arStart['cr']
            user_data.sensf = arStart['sens'][0]['sensitivity']
            predictionWindow.userData = user_data
            predictionWindow.plot = plotOption

            # Set minute index
            subset.index = (subset.index - subset.index[0]).seconds / 60
            predictionWindow.data = subset
            logger.debug(subset)
            if checkData.check_window(subset, user_data):
                res = check.checkAndPlot(predictionWindow)
                if res is not None:
                    results.append(res)

        startTime += delta  # delta determines the time between two predictions
    logger.debug("length of result " + str(len(results)))
    return results


