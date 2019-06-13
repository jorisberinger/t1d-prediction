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
from data.dataObject import DataObject

from matplotlib import gridspec, pyplot as plt

logger = logging.getLogger(__name__)

coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data_17_3-4.csv"
db_path = path + 'data/tinydb/db16.json'

# filename = path + "data/csv/data-o3.csv"

def main():
    logger.info("Start Main!")
    create_plots: bool = False  # Select True if you want a plot for every prediction window

    # SET USER DATA
    user_data = UserData(bginitial = 100.0, cratio = 5, idur = 4, inputeeffect = None, sensf = 41, simlength = 13,
                         predictionlength = 180, stats = None)

    # Get Database
    db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))

    logging.info("length of db: {}".format(len(db)))

    #all = db.all()

    with_result = db.search(where('result').exists())

    outliers = list(filter(lambda item: any(list(map(lambda result: abs(result['errors'][0]) > 100, item['result']))), with_result))

    logging.info("number of outliers: {}".format(len(outliers)))

    list(map(plot, outliers))
   
    exit()

    logging.info("results with optimizer: {} ".format(len(list(filter(lambda x: any(list(map(lambda y: 'Optimizer' in y['predictor'], x['result']))), with_result)))))
    for item in with_result:
        item['result'] = list(filter(lambda x: 'Optimizer' not in x['predictor'], item['result']))
        db.write_back([item])

    db.storage.flush()

    exit()


    logging.info("with result: {}".format(len(with_result)))

    results = list(map(lambda x: x['result'], with_result)) 

    seven = list(filter(lambda x : len(x) == 7, results))

    logging.info("with 7 results: {}".format(len(seven)))

    le = pd.Series(list(map(len, with_result)))
    logging.info(le.describe())

    exit()

    



    


    # Filter Items with LSMT Result

    lstm_result = list(filter(check_lstm, with_result))

    logging.info("number of results with lstm {}".format(len(lstm_result)))

    for item in lstm_result:
        item['result'] = list(filter(lambda x: x['predictor'] != 'LSTM Predictor', item['result']))
        db.write_back([item])

    db.storage.flush()

    exit()

    # with_result = db.search(where('doc_id') in list(range(19650, 19700)))
    # res = list(map(lambda x: db.get(doc_id=x),range(19600,19700)))
    # res = list(filter(lambda x: (x is not None), res))
    # logging.debug(len(res))

    #list(map(plot, res))
    # doc_ids = list(map(lambda x: x.doc_id, res))

    # db.remove(doc_ids=doc_ids)
    # db.storage.flush()
    # logging.info("length of db: {}".format(len(db)))
    # exit()

    # exit()
    outliers = list(filter(lambda x: x['result'][0]['errors'][0] > 75, with_result))
    logging.debug(len(outliers))
    # logging.debug(doc_ids)

    list(map(plot, outliers))

    logging.info('end')


    exit()
    for item in db:
        print(item)
        for x in item:
            print(x)
        exit()

def check_lstm(item):
    results = item['result']
    predictors = list(map(lambda x: x['predictor'], results))
    return 'LSTM Predictor' in predictors

def plot(item):
    logging.info('plotting {}'.format(item.doc_id))
    dataObject = DataObject.from_dict(item)
    cgm = dataObject.data['cgmValue']
    if max(cgm) > 600 or max(cgm) < 0:
        logging.info(max(cgm))
        logging.info("something wrong")

    dataObject.data['cgmValue'].plot()
    plt.scatter(dataObject.data['cgmValue_original'].index, dataObject.data['cgmValue_original'])
    
    plt.savefig('{}results/outliers/doc-{}.png'.format(path,item.doc_id))
    plt.close()
    logging.info("end")



def detect_outliers(item):
    dataObject = DataObject.from_dict(item)
    cgm = dataObject.data['cgmValue']
    m = max(cgm)
    mi = min(cgm)
    if m > 600 or m < 0 or mi < 20 or mi > 400:
        logging.debug(dataObject.start_time)
        logging.debug(m)
        return True
    return False


def check_outliers(item):
    return any(list(map(check_result, item['result'])))

def check_result(result):
    logging.info("result: {}".format(result))
    return abs(result['errors'][0]) > 100


if __name__ == "__main__":
    main()