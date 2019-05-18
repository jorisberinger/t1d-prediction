import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from keras import optimizers
import logging
import coloredlogs
import json
from sklearn.model_selection import train_test_split
from tinydb import TinyDB, JSONStorage, where
from tinydb.middlewares import CachingMiddleware
import os
from tensorflow.python.client import device_lib
from matplotlib import gridspec, pyplot as plt
coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

logging.info("Yeah")

### CHECK GPU
print(tf.test.gpu_device_name())
print(device_lib.list_local_devices())

### LOAD DATABASE
path = os.getenv('T1DPATH', '../')
db_path = path + 'data/tinydb/db1.json'
db = TinyDB(db_path, storage = CachingMiddleware(JSONStorage))
print("database loaded with {} entries".format(len(db)))

### PREPARE DATA
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


def load_data():
    logging.info("START execution")
    logging.info("Get all elements from DB")
    all_items = db.all()
    #all_items = all_items[:100]
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
  
  

df = load_data()


logging.info("Split into features and labels")
features = np.empty((len(df), 120,4))
labels = np.empty((len(df), 37))
for i, item in enumerate(df):
    features[i] = item.values[:600:5]
    labels[i] = item['cgmValue'].values[600::5]
    
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, shuffle = True)

#x_train = np.reshape(x_train.values, (x_train.shape[0], x_train.shape[1], 1))
#x_test = np.reshape(x_test.values, (x_test.shape[0], x_test.shape[1], 1)) 

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

logging.info("Machine Learning Start")
# define model
with tf.device('/CPU:0'):
    model = Sequential()
    model.add(LSTM(120, input_shape = (120, 4)))
    #model.add(CuDNNLSTM(600, input_shape = (600, 4), return_sequences= True))
    #model.add(Dropout(0.5))
    #model.add(CuDNNLSTM(120, return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(CuDNNLSTM(120))
    #model.add(Dropout(0.5))
    model.add(Dense(37))
    model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy'])

    model.summary()
    
    
    # fit model

    history = model.fit(x_train, y_train, epochs = 10, batch_size = 64 , validation_data = (x_test, y_test))
    # make a one step prediction out of sample
    # x_input = np.array([9, 10]).reshape((1, n_input, n_features))
    # yhat = model.predict(x_input, verbose=0)
    # print(yhat)

    test_acc = model.evaluate(x_test, y_test)

    print('Test accuracy:', test_acc)
    model.save(path+'model.h5') 

logging.info("END")

