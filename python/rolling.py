
import json
import logging
import os
import random
from datetime import timedelta, datetime

import pandas as pd
from tinydb import TinyDB, where

import check
from Classes import UserData
from PredictionWindow import PredictionWindow
from data import checkData
import checkOptimizer
from data.dataObject import DataObject

logger = logging.getLogger(__name__)
path = os.getenv('T1DPATH', '../')


# make rolling prediction and call checkWindow for every data window
def rolling(db: TinyDB, user_data: UserData, plotOption: bool):
    check_diretories()
    predictionWindow = PredictionWindow()

    results = []
    prediction_carb_optimized = []
    i = 0
    loop_start = datetime.now()

    # create random iterator
    elements = db.search(~where('result').exists())
    #elements = list(filter(lambda x: len(x['result']) < 10, elements))
    # elements = filter_elements(db.search(where('result').exists()))

    logging.info("number of unprocessed items {}".format(len(elements)))
    random.shuffle(elements)
    #elements = [db.get(doc_id=5748)]
    #elements = []
    for item in elements:
        # Break out of loop if enough results or it takes too long
        if len(results) > 10000 or \
                (datetime.now() - loop_start).seconds > 60 * 1:
            break
        logger.info("#:{} \t #R:{}\tdoc_id: {}".format(i, len(results), item.doc_id))
        # Get element
        data_object = DataObject.from_dict(item)

        predictionWindow.startTime = data_object.start_time
        predictionWindow.endTime = data_object.end_time
        # select data for this window
        predictionWindow.data = data_object.data
        predictionWindow.events = pd.concat([data_object.carb_events, data_object.basal_events, data_object.bolus_events])


        predictionWindow.userData = user_data
        predictionWindow.plot = plotOption


        # prediction_carb_optimized.append(checkOptimizer.check(predictionWindow))

        if checkData.check_window(predictionWindow.data, user_data):
            try:
                res = check.check_and_plot(predictionWindow, item)
                if res is not None:
                    results.append(res)
                    if 'result' in item:
                        item['result'] = item['result'] + res
                        db.write_back([item])
                    else:
                        db.update({'result': res}, doc_ids=[item.doc_id])
                    db.storage.flush()
            except:
                pass
    logger.info("length of result {}".format(len(results)))
    # save all prediction carb optimized values to a json file
    to_file(prediction_carb_optimized)

    return results

def filter_elements(elements: []) -> []:
    logging.info("in filter")
    fe = list(filter(lambda x: x['result'].pop(1)['errors'][0] > 100, elements))
    return fe

def to_file(arr):
    with open(path + "results/prediction_carbs.json", 'w+') as file:
        file.write(json.dumps(arr))


def check_diretories():
    directory = os.path.dirname(path + "results/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + '/plots/'):
        os.makedirs(directory + '/plots/')
