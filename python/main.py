import cProfile
import logging
import os
import time

import coloredlogs
import pandas as pd
from tinydb import TinyDB, JSONStorage, where
from tinydb.middlewares import CachingMiddleware
import config
import analyze
import gifmaker
import rolling
from Classes import UserData
from autotune import autotune
from data import readData, convertData
from data.dataPrep import add_gradient
import main_Prep as prep
from autotune.autotune_runner import run_autotune

logger = logging.getLogger(__name__)
coloredlogs.install(level = logging.INFO , fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')
path = os.getenv('T1DPATH', '../')

# SET INPUT FILE PATH
filepath = path + config.DATA_CONFIG['csv_input_path']
db_path = path + config.DATA_CONFIG['database_path']


def main():
    logger.info("Start Main!")

    # SET USER DATA
    user_data = UserData(bginitial = 100.0, cratio = 1, idur = 4, inputeeffect = None, sensf = 1, simlength = 13,
                         predictionlength = 180, stats = None)

    # Calculate Insulin Sensitivity factor and Carbohydrate Ratio with autotune
    #run_autotune(filepath) 

    # Get Database
    logging.debug("Loading database...")
    db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
    logging.info("Loaded database from {} with {} items".format(os.path.abspath(db_path),len(db)))
    logging.info("Valid items: {}".format(len(db.search(where('valid') == True))))

    # Load data from csv into database
    # prep.main(db, filepath)

    # exit(0)
    # MAKE A ROLLING PREDICTION
    logger.info("Start Prediction")
    prediction_result = rolling.rolling(db, user_data)
    logger.info("Finished prediction")
    
 
    # CREATE PLOTS FOR ANALYSE SUMMARY
    logger.info("creating error plots..")
    analyze.createErrorPlots(db)

    logger.info("Main finished!")


if __name__ == '__main__':
    start_time = time.process_time()
    profile = True
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    main()
    if profile:
        pr.disable()
        pr.dump_stats(path + "results/profiler/profile")
    logger.info(str(time.process_time() - start_time) + " seconds")
