
import json
import logging
import os
from datetime import timedelta, datetime

import pandas as pd

import check
from Classes import UserData
from PredictionWindow import PredictionWindow
from data import checkData
import checkOptimizer

logger = logging.getLogger(__name__)
path = os.getenv('T1DPATH', '../')


# make rolling prediction and call checkWindow for every data window
def rolling(data: pd.DataFrame, delta: pd.Timedelta, user_data: UserData, autotune_res: dict, plotOption: bool):
    check_diretories()
    predictionWindow = PredictionWindow()

    # select starting point as first data point, always start at 15 minute intervals
    startTime = data.index[0]
    startTime = startTime.replace(minute = data.index[0].minute)

    endTime = data.index[len(data) - 1]
    predictionWindow.endTime = data.index[len(data) - 1]
    results = []
    prediction_carb_optimized = []
    i = 0
    loop_start = datetime.now()
    # loop through the data
    while startTime < endTime - timedelta(hours = user_data.simlength) \
            and len(results) < 1000000 \
            and (datetime.now() - loop_start).seconds < 60 * 60 * 14:  # TODO use a global variable
        logger.info("#:{} \t #R:{}".format(i, len(results)))
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
            # optimize test window
            prediction_carb_optimized.append(checkOptimizer.check(predictionWindow))

            if checkData.check_window(subset, user_data):
                res = check.check_and_plot(predictionWindow)
                if res is not None:
                    results.append(res)

        startTime += delta  # delta determines the time between two predictions
    logger.info("length of result {}".format(len(results)))
    # save all prediction carb optimized values to a json file
    to_file(prediction_carb_optimized)
    return results


def to_file(arr):
    with open(path + "results/prediction_carbs.json", 'w+') as file:
        file.write(json.dumps(arr))


def check_diretories():
    directory = os.path.dirname(path + "results/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + '/plots/'):
        os.makedirs(directory + '/plots/')
