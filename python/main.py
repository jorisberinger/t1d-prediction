import cProfile
import logging
import os
import time

import coloredlogs
import pandas as pd
from tinydb import TinyDB, JSONStorage, where
from tinydb.middlewares import CachingMiddleware

import analyze
import gifmaker
import rolling
from Classes import UserData
from autotune import autotune
from data import readData, convertData
from data.dataPrep import add_gradient

logger = logging.getLogger(__name__)

coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data_17_3-4.csv"
db_path = path + 'data/tinydb/db1.json'

# filename = path + "data/csv/data-o3.csv"

def main():
    logger.info("Start Main!")
    create_plots: bool = False  # Select True if you want a plot for every prediction window

    # SET USER DATA
    user_data = UserData(bginitial = 100.0, cratio = 5, idur = 4, inputeeffect = None, sensf = 41, simlength = 13,
                         predictionlength = 180, stats = None)

    # Get Database
    db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))

    #add_gradient(db)


    # MAKE A ROLLING PREDICTION
    logger.info("Start Prediciton")
    prediction_result = rolling.rolling(db, user_data, create_plots)
    logger.info("Finished prediction")

    # ANALYSE PREDICTION RESULTS
    #summary, all_data = analyze.getSummary(db)

    # CREATE PLOTS FOR ANALYSE SUMMARY
    #analyze.createErrorPlots(summary, all_data)


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
