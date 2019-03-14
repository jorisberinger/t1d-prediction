import logging
import os
from datetime import timedelta

import pandas as pd

import check
from Classes import PredictionWindow, UserData
from data import checkData

logger = logging.getLogger(__name__)
path = os.getenv('T1DPATH', '../')


# make rolling prediction and call checkWindow for every data window
def rolling(data: pd.DataFrame, delta: pd.Timedelta, user_data: UserData, autotune_res: dict, plotOption: bool):
    checkDiretories()
    predictionWindow = PredictionWindow()

    # select starting point as first data point, always start at 15 minute intervals
    startTime = data.index[0]
    startTime = startTime.replace(minute = data.index[0].minute)

    endTime = data.index[len(data) - 1]
    predictionWindow.endTime = data.index[len(data) - 1]
    results = []
    orders = []
    i = 0
    # loop through the data
    while startTime < endTime - timedelta(hours = user_data.simlength) and len(results) < 20: # TODO use a global variable
        logger.info("#" + str(i))
        logger.info("#r " + str(len(results)))
        i += 1
        predictionWindow.startTime = startTime
        predictionWindow.endTime = startTime + timedelta(hours = user_data.simlength)
        # select data for this window
        subset = data.loc[startTime <= data.index]
        subset = subset.loc[predictionWindow.endTime >= subset.index]

        # Set Sensitivity factor and CarbRatio
        if len(subset) > 0:

            arStart = autotune_res[subset.date.values[0]]
            # arEnd = autotune_res[subset.date.values[len(subset.date.values) -1]]  # TODO use second day if overlapping
            user_data.cratio = arStart['cr']
            user_data.sensf = arStart['sens'][0]['sensitivity']
            predictionWindow.userData = user_data
            predictionWindow.plot = plotOption

            # Set minute index
            subset.index = (subset.index - subset.index[0]).seconds / 60
            predictionWindow.data = subset
            # logger.debug(subset)
            if checkData.check_window(subset, user_data):
                res, order = check.checkAndPlot(predictionWindow)
                if res is not None:
                    orders.append(order)
                    results.append(res)

        startTime += delta  # delta determines the time between two predictions
    logger.debug("length of result " + str(len(results)))
    logger.info("Orders")
    logger.info(orders)
    return results


def checkDiretories():
    directory = os.path.dirname(path + "results/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + '/plots/'):
        os.makedirs(directory + '/plots/')
