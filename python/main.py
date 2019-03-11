import cProfile
import logging
import os
import time

import coloredlogs
import pandas as pd

import analyze
import gifmaker
import rolling
from Classes import UserData
from autotune import autotune
from data import readData, convertData

logger = logging.getLogger(__name__)

coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data_17_5.csv"


# filename = path + "data/csv/data-o3.csv"

def main():
    # SELECT OPTIONS
    run_autotune: bool = False  # Select True if autotune should run. If data set has been run before, set to False to improve speed.
    create_plots: bool = True  # Select True if you want a plot for every prediction window

    # SET USER DATA
    user_data = UserData(bginitial = 100.0, cratio = 5, idur = 4, inputeeffect = None, sensf = 41, simlength = 11,
                         predictionlength = 60, stats = None)

    logger.info("Start Main!")

    # LOAD DATA
    data = readData.read_data(filename)

    # CLEAN UP DATA FRAME
    data = convertData.convert(data)

    # INTERPOLATE CGM MEASURES
    data = convertData.interpolate_cgm(data)

    # GET SENSITIVITY FACTOR AND CARBRATIO FOR EVERY DAY
    logger.info("Run Autotune? " + str(run_autotune))
    if run_autotune:
        autotune_res = autotune.runAutotune(data)
    else:
        autotune_res = autotune.getAllSensAndCR(data)

    # DROP DATE AND TIME STRINGS
    # data = convertData.drop_date_and_time(data)
    # MAKE A ROLLING PREDICTION
    logger.debug("Run Prediciton")
    prediction_result = rolling.rolling(data, pd.Timedelta('15 minutes'), user_data, autotune_res, create_plots)
    logger.info("Finished prediction")

    # ANALYSE PREDICTION RESULTS
    #summary = analyze.getSummary(prediction_result)

    # CREATE PLOTS FOR ANALYSE SUMMARY
    #analyze.createErrorPlots(summary)

    # CREATE A GIF OUT OF THE PREDICTION PLOTS
    #if create_plots:
    #    gifmaker.makeGif(path + "results/plots/", data)

    logger.info("Main finished!")


if __name__ == '__main__':
    start_time = time.process_time()
    profile = False
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    main()
    if profile:
        pr.disable()
        pr.dump_stats(path + "results/profiler/profile")
    logger.info(str(time.process_time() - start_time) + " seconds")
