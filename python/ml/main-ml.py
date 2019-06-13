import coloredlogs
import logging 
import os
from tinydb import TinyDB, JSONStorage, where
from tinydb.middlewares import CachingMiddleware
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import CuDNNLSTM, Dropout, Dense
from keras.models import Sequential
coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
db_path18 = path + 'data/tinydb/db18.json'
db_path17 = path + 'data/tinydb/db1.json'


def main():
    logging.info("Starting machine learning main")
    db18 = TinyDB(db_path18, storage = CachingMiddleware(JSONStorage))
    logging.info("database loaded with {} entries".format(len(db18)))
    db17 = TinyDB(db_path17, storage = CachingMiddleware(JSONStorage))
    logging.info("database loaded with {} entries".format(len(db17)))
    df18 = load_data(db18)
    df17 = load_data(db17)
    train_model(df17)
    logging.info("Done")



def train_model(df):
    logging.info("Split into features and labels")
    features = np.empty((len(df), 120,4))
    labels = np.empty((len(df), 37))
    for i, item in enumerate(df):
        features[i] = item.values[:600:5]
        labels[i] = item['cgmValue'].values[600::5]
        
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, shuffle = True)
    with tf.device('/GPU:0'):
        model = Sequential()
        #model.add(CuDNNLSTM(120, input_shape = (120, 4)))
        model.add(CuDNNLSTM(120, input_shape = (120, 4), return_sequences= True))
        model.add(Dropout(0.5))
        model.add(CuDNNLSTM(40))
        #model.add(Dropout(0.5))
        model.add(Dense(37))
        model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy', 'mae'])

        model.summary()
        
        
        # fit model

        history = model.fit(x_train, y_train, epochs = 1000, batch_size = 256 , validation_data = (x_test, y_test))
        # make a one step prediction out of sample
        # x_input = np.array([9, 10]).reshape((1, n_input, n_features))
        # yhat = model.predict(x_input, verbose=0)
        # print(yhat)

        test_acc = model.evaluate(x_test, y_test)
        
        model.save('{}models/model17-2l-1000.h5'.format(path)) 

        logging.info('Test mae: {}'.format(test_acc[2]))


def convert_to_int(x):

    try:
        return float(x)
    except:
        logging.error("error {}".format(x))



def sort_index(item):
    item.index = list(map(lambda x: convert_to_int(x), item.index))
    return item.sort_index()

etypes = ['carb', 'bolus', 'basal']
def from_dict(di):
    data_object = {}
    # data_object['start_time'] = (pd.datetime.fromisoformat(di['start_time']))
    # data_object['end_time'] = (pd.datetime.fromisoformat(di['end_time']))
    data_object['data'] = (pd.DataFrame(json.loads(di['data'])))
    if 'result' in di:
        data_object['result'] = di['result']
    # for key in etypes:
    #    if key in di:
    #        df = pd.DataFrame(json.loads(di[key]))
    #        df.index = list(map(lambda x: float(x), df.index))
    #        data_object.__setattr__(key + '_events', df.sort_index())
    return data_object


def load_data(db):
    logging.info("START execution")
    logging.info("Get all elements from DB")
    all_items = db.all()
    logging.info("{} items found".format(len(all_items)))
    logging.info("Convert items to data_objects")
    data_objects = list(map(lambda x: from_dict(x), all_items))
    logging.info("Extract cgmValues")
    feature_list = list(map(lambda x: x['data'][['cgmValue', 'basalValue', 'bolusValue', 'mealValue']].fillna(0), data_objects))
    logging.info("Set Index and Sort")
    subset = list(map(sort_index, feature_list))
    for item in subset:
      item['cgmValue'] /= 500
    #subset = subset[:500]
    logging.info("Convert to Dataframe")
    return subset
  




if __name__ == "__main__":
    main()