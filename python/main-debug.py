import cProfile
import logging
import os
import time

import coloredlogs
import pandas as pd
import numpy as np
from tinydb import TinyDB, JSONStorage, where 
from tinydb.middlewares import CachingMiddleware
from tinydb.operations import delete

# import analyze
# import gifmaker
# import rolling
# from Classes import UserData
# from autotune import autotune
# from data import readData, convertData
# from data.dataPrep import add_gradient
# from data.dataObject import DataObject

from matplotlib import gridspec, pyplot as plt
from collections import Counter

import keras

logger = logging.getLogger(__name__)

coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
#filename = path + "data/csv/csv_4.csv"
# db_path = path + 'data/tinydb/db4p.json'
db_path = path + 'data/tinydb/dbtest2.json'



    logger.info("Start debugging!")
    logger.info("db: {}".format(db_path.split('/')[-1]))
    create_plots = False  # Select True if you want a plot for every prediction window

    # SET USER DATA
    user_data = UserData(bginitial = 100.0, cratio = 5, idur = 4, inputeeffect = None, sensf = 41, simlength = 13,
                         predictionlength = 180, stats = None)





    

#     # Get Database
#     db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))



#     logging.info("length of db: {}".format(len(db)))
#     logging.info("Valid examples: {}".format(len(db.search(where('valid') == True))))
#     logging.info("With result: {}".format(len(db.search(where('result').exists()))))

    
    
#     with_result = db.search(where('result').exists())
#     get_predictor_count(with_result)




#     exit()
#     all_items = list(filter(lambda x: x['id'] == '29032313', with_result)) # Train - 3p
#     get_predictor_count(all_items)

#     exit()
#     get_predictor_count(with_result)
#     cleaned = list(map(lambda x: clean_result(x, 'Mean'), with_result))
#     get_predictor_count(cleaned)
#     cleaned = list(map(lambda x: clean_result(x, 'LSTM'), with_result))
#     get_predictor_count(cleaned)
#     cleaned = list(map(lambda x: clean_result(x, '[15, 30,'), with_result))
#     get_predictor_count(cleaned)


#     exit()
#     with_result = db.search(where('result').exists())
#     all_items = list(filter(lambda x: x['id'] == '82923830', with_result)) # Train - 3p
#     # all_items = list(filter(lambda x: x['id'] == '27283995', with_result))  # Test patient - 2p
#     p_result = list(filter(lambda x: any(list(map(lambda y: "LSTM" in y['predictor'], x['result']))) , all_items))
#     logging.info("error res {}".format(len(p_result)))
#     p_cleaned = list(map(lambda x: clean_result(x, '5000'), p_result))

#     db.write_back(p_cleaned)
#     db.storage.flush()

#     exit()
#     aa = db.search(where('lstm-test-result').exists())
#     logging.info("found {} items with lstm test result".format(len(aa)))
    
#     all = db.search(where('valid') == True)
#     #s = pd.Series(list(map(lambda x: x['id'], all)))
#     #list(map(lambda id: print("id {} has {} items".format(id, len(list(filter(lambda x: x['id'] == id, all))))), s.unique()))

    
#     with_result = db.search(where('result').exists())
#     lstm_result = list(filter(lambda x: any(list(map(lambda y: "LSTM" in y['predictor'], x['result']))) , with_result))
#     logging.info("lstm res {}".format(len(lstm_result)))
#     lstm_cleaned = list(map(clean_lstm, lstm_result))

#     db.write_back(lstm_cleaned)
#     db.storage.flush()
#     with_result = db.search(where('result').exists())
#     lstm_result = list(filter(lambda x: any(list(map(lambda y: "LSTM" in y['predictor'], x['result']))) , with_result))

#     logging.info("lstm res {}".format(len(lstm_result)))

#     exit()

#     all = db.search(where('valid') == True)
#     s = pd.Series(list(map(lambda x: x['id'], all)))
#     list(map(lambda id: print("id {} has {} items".format(id, len(list(filter(lambda x: x['id'] == id, all))))), s.unique()))

#     exit()
#     get_arima_order_summary(db)

#     exit()

#     with_result = db.search(where('result').exists())
#     arima_result = list(filter(lambda x: any(list(map(lambda y: y['predictor'] == 'Arima Predictor', x['result']))) , with_result))

#     logging.info("arima results: {}".format(len(arima_result)))

   
    


#     logging.info("length of db: {}".format(len(db)))

#     #all = db.all()

    

#     outliers = list(filter(lambda item: any(list(map(lambda result: abs(result['errors'][0]) > 100, item['result']))), with_result))

#     logging.info("number of outliers: {}".format(len(outliers)))

#     list(map(plot, outliers))
   
#     exit()

#     logging.info("results with optimizer: {} ".format(len(list(filter(lambda x: any(list(map(lambda y: 'Optimizer' in y['predictor'], x['result']))), with_result)))))
#     for item in with_result:
#         item['result'] = list(filter(lambda x: 'Optimizer' not in x['predictor'], item['result']))
#         db.write_back([item])

#     db.storage.flush()

#     exit()


#     logging.info("with result: {}".format(len(with_result)))

#     results = list(map(lambda x: x['result'], with_result)) 

#     seven = list(filter(lambda x : len(x) == 7, results))

#     logging.info("with 7 results: {}".format(len(seven)))

#     le = pd.Series(list(map(len, with_result)))
#     logging.info(le.describe())

#     exit()

    



    


#     # Filter Items with LSMT Result

#     lstm_result = list(filter(check_lstm, with_result))

#     logging.info("number of results with lstm {}".format(len(lstm_result)))

#     for item in lstm_result:
#         item['result'] = list(filter(lambda x: x['predictor'] != 'LSTM Predictor', item['result']))
#         db.write_back([item])

#     db.storage.flush()

#     exit()

#     # with_result = db.search(where('doc_id') in list(range(19650, 19700)))
#     # res = list(map(lambda x: db.get(doc_id=x),range(19600,19700)))
#     # res = list(filter(lambda x: (x is not None), res))
#     # logging.debug(len(res))

#     #list(map(plot, res))
#     # doc_ids = list(map(lambda x: x.doc_id, res))

#     # db.remove(doc_ids=doc_ids)
#     # db.storage.flush()
#     # logging.info("length of db: {}".format(len(db)))
#     # exit()

#     # exit()
#     outliers = list(filter(lambda x: x['result'][0]['errors'][0] > 75, with_result))
#     logging.debug(len(outliers))
#     # logging.debug(doc_ids)

#     list(map(plot, outliers))

#     logging.info('end')


#     exit()
#     for item in db:
#         print(item)
#         for x in item:
#             print(x)
#         exit()

# def check_lstm(item):
#     results = item['result']
#     predictors = list(map(lambda x: x['predictor'], results))
#     return 'LSTM Predictor' in predictors

# def plot(item):
#     logging.info('plotting {}'.format(item.doc_id))
#     dataObject = DataObject.from_dict(item)
#     cgm = dataObject.data['cgmValue']
#     if max(cgm) > 600 or max(cgm) < 0:
#         logging.info(max(cgm))
#         logging.info("something wrong")

#     dataObject.data['cgmValue'].plot()
#     plt.scatter(dataObject.data['cgmValue_original'].index, dataObject.data['cgmValue_original'])
    
#     plt.savefig('{}results/outliers/doc-{}.png'.format(path,item.doc_id))
#     plt.close()
#     logging.info("end")



# def detect_outliers(item):
#     dataObject = DataObject.from_dict(item)
#     cgm = dataObject.data['cgmValue']
#     m = max(cgm)
#     mi = min(cgm)
#     if m > 600 or m < 0 or mi < 20 or mi > 400:
#         logging.debug(dataObject.start_time)
#         logging.debug(m)
#         return True
#     return False


# def check_outliers(item):
#     return any(list(map(check_result, item['result'])))

# def check_result(result):
#     logging.info("result: {}".format(result))
#     return abs(result['errors'][0]) > 100

# def get_arima_order_summary(db):
#     order  = list(map(lambda x: x['features'], db.search(where('features').exists())))
#     cnt = Counter()
#     for ord in order:
#         cnt[str(ord)] +=1


#     list(map(lambda x: print(x), cnt.most_common()))


# def get_features_summary(db: TinyDB):
#     logging.info("Getting features summary")
#     features = list(map(lambda x: x['features-90'], db.search(where('features-90').exists())))
#     logging.info("found {} results".format(len(features)))


# def clean_result(item, string):
#     logging.debug("results before: {}".format(list(map(lambda x: x['predictor'], item['result']))))
#     item['result'] = list(filter(lambda x: string not in x['predictor'], item['result']))
#     logging.debug("results after: {}".format(list(map(lambda x: x['predictor'], item['result']))))
#     return item


# def get_predictor_count(all_items: []):
#     r = list(map(lambda x: x['result'], all_items))
#     predictors = []
#     for item in r:
#         for pred in item:
#             predictors.append(pred['predictor'])
#     preds = pd.Series(predictors)
#     logging.info(preds.value_counts())


if __name__ == "__main__":
    main()