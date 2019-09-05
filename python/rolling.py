
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
    # elements = db.search(where('result').exists() & (where('valid') == True))
    #elements = db.search((where('valid') == True) & where('lstm-test-result').exists() & where('error-arima-result').exists())
    #elements = db.search((where('valid') == True) & where('optimizer-error-prediction-1000').exists())
    # elements = db.search((where('valid') == True))

    # filter elements by day of month, relevant for db with one patient
    #elements = list(filter(check_time_train, elements))
    #elements = list(filter(check_time_test, elements))

    # filter elements by patient id, relevant for db with more than 1 patient
    # elements = list(filter(lambda x: x['id'] == '82923830', elements))  # TRAIN patient
    # elements = list(filter(lambda x: x['id'] == '27283995', elements))  # Test patient
    # elements = list(filter(lambda x: x['id'] == '29032313', elements))  # Third patient

    elements = [db.get(doc_id=28524)]

    #elements = list(filter(lambda x: any(list(map(lambda y: abs(y['errors'][0]) > 70, x['result']))), elements))

    #random.shuffle(elements)
    logging.info("number of unprocessed items {}".format(len(elements)))

    last_save = 0
    for item in elements:
        # Break out of loop if enough results or it takes too long
        if len(results) >= config.PREDICTION_CONFIG['max_number_of_results'] or \
                (datetime.now() - loop_start).seconds > config.PREDICTION_CONFIG['runtime_in_minutes'] * 60:
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
        predictionWindow.features_90 = data_object.features_90
        PredictionWindow.lstm_result = data_object.lstm_result
        PredictionWindow.error_result = data_object.error_result

        # prediction_carb_optimized.append(checkOptimizer.check(predictionWindow))

        if checkData.check_window(predictionWindow.data, user_data):
            # Call to Predictors
            res, order, features = check.check_and_plot(predictionWindow, item)
            # Write result back into db
            if res is not None:
                results.append(res)
                if 'result' in item:
                    item['result'] = item['result'] + res
                    db.write_back([item])
                else:
                    db.update({'result': res}, doc_ids=[item.doc_id])
            if features is not None:
                if 'features-90' in item:
                    item['features-90'] = features
                    db.write_back([item])
                else:
                    db.update({'features-90': features}, doc_ids=[item.doc_id])
            if order is not None:
                if 'features' in item:
                    item['features'] = order
                    db.write_back([item])
                else:
                    db.update({'features': order}, doc_ids=[item.doc_id])
        
        if len(results) > 200 + last_save:
            last_save = len(results)
            #db.storage.flush()       
    #db.storage.flush()
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

def check_time_test(item):
    time = pd.Timestamp(item['start_time'])
    if time.day in [5,12]:
        return True
    if time.day in [4,11] and time.hour > 10:
        return True
    if time.day in [6,13] and time.hour < 14:
        return True
    return False

def check_time_train(item):
    time = pd.Timestamp(item['start_time'])
    return time.day not in [4,5,6,11,12,13]
