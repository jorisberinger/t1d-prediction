
import json
import logging
import os
import random
from datetime import timedelta, datetime
import config
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
def rolling(db: TinyDB, user_data: UserData):
    check_directories()
    predictionWindow = PredictionWindow()

    results = []
    prediction_carb_optimized = []
    i = 0
    loop_start = datetime.now()

    # create random iterator over valid items without a result
    elements = db.search(~where('result').exists() & (where('valid') == True))
    random.shuffle(elements)
    logging.info("number of unprocessed items {}".format(len(elements)))

    for item in elements:
        # Break out of loop if enough results or it takes too long
        if len(results) >= config.PREDICTION_CONFIG['max_number_of_results'] or \
                (datetime.now() - loop_start).seconds > config.PREDICTION_CONFIG['runtime_in_minutes']:
            break
        logger.info("#:{} \t #R:{}\tdoc_id: {}".format(i, len(results), item.doc_id))
        # Get element
        data_object = DataObject.from_dict(item)

        predictionWindow.startTime = data_object.start_time
        logger.info(predictionWindow.startTime.isoformat())
        predictionWindow.endTime = data_object.end_time
        # select data for this window
        predictionWindow.data = data_object.data
        predictionWindow.data_long = data_object.data_long
        predictionWindow.events = pd.concat([data_object.carb_events, data_object.basal_events, data_object.bolus_events])
        predictionWindow.userData = user_data
        predictionWindow.plot = config.PREDICTION_CONFIG['create_plots']

        # prediction_carb_optimized.append(checkOptimizer.check(predictionWindow))

        if checkData.check_window(predictionWindow.data, user_data):
            # Call to Predictors
            res, order = check.check_and_plot(predictionWindow, item)
            # Write result back into db
            if res is not None:
                results.append(res)
                if 'result' in item:
                    item['result'] = item['result'] + res
                    db.write_back([item])
                else:
                    db.update({'result': res}, doc_ids=[item.doc_id])
            if order is not None:
                if 'features' in item:
                    item['features'] = order
                    db.write_back([item])
                else:
                    db.update({'features': order}, doc_ids=[item.doc_id])
        db.storage.flush()       

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


def check_directories():
    directory = os.path.dirname(path + "results/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + '/plots/'):
        os.makedirs(directory + '/plots/')
