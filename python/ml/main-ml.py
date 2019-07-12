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
from keras.layers import CuDNNLSTM, Dropout, Dense, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
db_path18 = path + 'data/tinydb/db18.json'
db_path17 = path + 'data/tinydb/db1.json'
db_path = path + 'data/tinydb/dbtest2.json'

# Different configuarations of Features
# ['cgmValue', 'basalValue', 'bolusValue', 'mealValue', 'feature-90']
configurations = [  {'name':'cgm', 'columns':[0]},
                        {'name':'cgm, insulin', 'columns':[0,1, 2]},
                        {'name':'cgm, insulin, carbs', 'columns':[0, 1, 2, 3]},
                        {'name':'cgm, insulin, optimized', 'columns':[0, 1, 2, 4]},
                        {'name':'cgm, insulin, carbs, optimized', 'columns':[0, 1, 2, 3, 4]}                    
                    ]

def main():
    logging.info("Starting machine learning main")
    #db = TinyDB(db_path18, storage = CachingMiddleware(JSONStorage))
    #logging.info("database loaded with {} entries".format(len(db18)))
    #db17 = TinyDB(db_path17, storage = CachingMiddleware(JSONStorage))
    logging.info("Load Database")
    db = TinyDB(db_path, storage = CachingMiddleware(JSONStorage))
    logging.info("database loaded with {} entries".format(len(db)))
    #df18 = load_data(db18)
    #df17 = load_data(db17)
    logging.info("convert data into files")
    df = load_data_with_result(db)
    compare_features(df)
    logging.info("Done")


def compare_features(df):
    logging.info("Comparing model with different features")
    
    for configuration in configurations:
        configuration['number_features'] = len(configuration['columns'])
        # get features and labels for configuration
        features, labels = get_features_for_configuration(configuration, df)
        
        # split data into test and train data
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, shuffle = True, random_state=1)

        train_model_for_configuration(x_train, x_test, y_train, y_test, configuration)
        
        logging.info("Done with {}".format(configuration['name']))

    for configuration in configurations:
        logging.info("{} with mae {}".format(configuration['name'], configuration['mae']))


def get_features_for_configuration(configuration, df:pd.DataFrame):
    logging.info("Prepping for {}".format(configuration['name']))

    features = np.empty((len(df), 120, configuration['number_features']))
    all_features = np.empty((len(df), 120, 5))
    labels = np.empty((len(df), 37))
    for i, item in enumerate(df):
        all_features[i] = item.values[:600:5]
        labels[i] = item['cgmValue'].values[600::5]

    logging.info("feature shape {}".format(all_features.shape))
    features = all_features[:,:,configuration['columns']]
    logging.info("feature shape {}".format(features.shape))

    return features, labels

def train_model_for_configuration(x_train, x_test, y_train, y_test, configuration):

    with tf.device('/GPU:0'):
            # define Model
            model = Sequential()
            model.add(CuDNNLSTM(120, input_shape = (120, configuration['number_features']), return_sequences= True))
            model.add(Dropout(0.5))
            model.add(CuDNNLSTM(40, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(CuDNNLSTM(40))
            model.add(Dense(37))
            model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy', 'mae'])
            model.summary()
            #
            es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=100)
            # fit model
            history = model.fit(x_train, y_train, epochs = 1000, batch_size = 256 , validation_data = (x_test, y_test), callbacks=[es])
            test_acc = model.evaluate(x_test, y_test)
            model.save('{}models/test-1000-{}.h5'.format(path, configuration['name'])) 
            logging.info('Test mae: {}'.format(test_acc[2]))
            configuration['mae'] = test_acc[2]




def train_model(df):
    logging.info("Split into features and labels")
    features = np.empty((len(df), 120,5))
    labels = np.empty((len(df), 37))
    for i, item in enumerate(df):
        features[i] = item.values[:600:5]
        labels[i] = item['cgmValue'].values[600::5]
        
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, shuffle = True)
    with tf.device('/GPU:0'):
        model = Sequential()
        #model.add(CuDNNLSTM(120, input_shape = (120, 4)))
        model.add(CuDNNLSTM(120, input_shape = (120, 5), return_sequences= True))
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
        
        model.save('{}models/test.h5'.format(path)) 

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
    data_object['features-90'] = di['features-90']
    if 'result' in di:
        data_object['result'] = di['result']
    # for key in etypes:
    #    if key in di:
    #        df = pd.DataFrame(json.loads(di[key]))
    #        df.index = list(map(lambda x: float(x), df.index))
    #        data_object.__setattr__(key + '_events', df.sort_index())
    return data_object

def load_data_with_result(db):
    logging.info("START execution")
    logging.info("Get all elements from DB")
    all_items = db.search(where('features-90').exists())
    all_items = list(filter(lambda x: len(x['result']) == 5, all_items))
    logging.info("{} items found".format(len(all_items)))
    logging.info("Convert items to data_objects")
    data_objects = list(map(lambda x: from_dict(x), all_items))
    logging.info("Extract cgmValues")
    feature_list = list(map(get_feature_list , data_objects))
    logging.info("Set Index and Sort")
    subset = list(map(sort_index, feature_list))
    for item in subset:
      item['cgmValue'] /= 500
    #subset = subset[:500]
    logging.info("Convert to Dataframe")
    return subset

def get_feature_list(data_object):
    df = data_object['data'][['cgmValue', 'basalValue', 'bolusValue', 'mealValue']].fillna(0)
    df.index = list(map(float , df.index))
    df['features-90'] = 0
    for i, v in enumerate(range(0,600,15)):
        df.loc[v,'features-90'] = data_object['features-90'][i]
    return df

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