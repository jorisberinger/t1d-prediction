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

    l = db.search(where('result').exists() & (where('gradient-s') > 10))

    print(len(l))

    exit()
    for item in db:
        print(item)
        for x in item:
            print(x)
        exit()

    
if __name__ == "__main__":
    main()