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
from matplotlib import pyplot as plt
import keras
coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
db_path18 = path + 'data/tinydb/db18.json'
db_path17 = path + 'data/tinydb/db1.json'

db_path = path + 'data/tinydb/db3p.json'


model_path = path+'models/1p-gpu-2000-3l-cgm, insulin, carbs, optimized, tod.h5'
#model_path = path+'models/1p-cpu-5-3l-cgm, insulin, carbs, optimized, tod.h5'

# Different configuarations of Features
# ['cgmValue', 'basalValue', 'bolusValue', 'mealValue', 'feature-90', 'timeOfDay']
configurations = [      #{'name':'cgm', 'columns':[0]},
                        #{'name':'cgm, time of day', 'columns':[0, 5]},
                        #{'name':'cgm, insulin', 'columns':[0,1, 2]},
                        #{'name':'cgm, insulin, tod', 'columns':[0,1, 2, 5]},
                        #{'name':'cgm, insulin, carbs', 'columns':[0, 1, 2, 3]},
                        #{'name':'cgm, insulin, carbs, tod', 'columns':[0, 1, 2, 3, 5]},
                        #{'name':'cgm, insulin, optimized', 'columns':[0, 1, 2, 4]},
                        #{'name':'cgm, insulin, optimized, tod', 'columns':[0, 1, 2, 4, 5]},
                        #{'name':'cgm, insulin, carbs, optimized', 'columns':[0, 1, 2, 3, 4]},
                        {'name':'cgm, insulin, carbs, optimized, tod', 'columns':[0, 1, 2, 3, 4, 5]}                  
                    ]

def main():

    path = os.getenv('T1DPATH', '../')
    model_path = path+'models/mp-err-arima-1000-1l-cgm, insulin, carbs, optimized, tod.h5'
    model = keras.models.load_model(model_path)
    keras.utils.plot_model(model, to_file='model.png')
    exit()


    logging.info("Starting machine learning main")
    logging.info("Load Database")
    db = TinyDB(db_path, storage = CachingMiddleware(JSONStorage))
    logging.info("database loaded with {} entries".format(len(db)))

    logging.info("convert data into files")
    df = None
    # df = load_data_with_result(db)
    
    predictions = compare_features(df)
    save_predictions(db,predictions)
    logging.info("Done")


def compare_features(df):
    logging.info("Comparing model with different features")
    
    for configuration in configurations:
        configuration['number_features'] = len(configuration['columns'])
        # get features and labels for configuration
        features, labels, doc_ids = get_features_for_configuration(configuration, df)

       
        #features.tofile(path+'models/np-2p-test-features')
        #labels.tofile(path+'models/np-2p-test-labels')
        #doc_ids.tofile(path+'models/np-2p-test-docids')


        # ## 2p Train features
        # features = np.fromfile(path+'models/np-2p-features')
        # features = features.reshape((6178, 121, 6))
        # labels = np.fromfile(path+'models/np-2p-labels')
        # labels = labels.reshape((6178, 37))
        # doc_ids = np.fromfile(path+'models/np-2p-doc_ids')
        # doc_ids = doc_ids.reshape((6178,))

        # ## 2P Test features
        # features_test = np.fromfile(path+"models/np-2p-test-features")
        # features_test = features_test.reshape((998, 121, 6))
        # labels_test = np.fromfile(path+"models/np-2p-test-labels")
        # labels_test = labels_test.reshape((998, 37))
        # doc_ids_test = np.fromfile(path+"models/np-2p-test-docids")
        # doc_ids_test = doc_ids_test.reshape((998,))


        # ## 1P Train features
        # features = np.fromfile(path+"models/np-1p-features")
        # features = features.reshape((4506, 121, 6))
        # labels = np.fromfile(path+"models/np-1p-labels")
        # labels = labels.reshape((4506, 37))
        # ## 1P Test features
        # features_test = np.fromfile(path+"models/np-1p-test-features")
        # features_test = features_test.reshape((966, 121, 6))
        # labels_test = np.fromfile(path+"models/np-1p-test-labels")
        # labels_test = labels_test.reshape((966, 37))
        # doc_ids_test = np.fromfile(path+"models/np-1p-test-docids")
        # doc_ids_test = doc_ids_test.reshape((966,))



        # features_test = features_test[0:10]
        # doc_ids_test = doc_ids_test[0:10]


        # split data into test and train data
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, shuffle = True, random_state=1)

        model = train_model_for_configuration(x_train, x_test, y_train, y_test, configuration)
        #model = keras.models.load_model(model_path)
        logging.info("Done with {}".format(configuration['name']))

        predictions = predict_with_model(features_test, labels_test, doc_ids_test, model)

        return predictions
        #plot_lstm_predictions(model, x_test, y_test)

    for configuration in configurations:
        logging.info("{} with mae {}".format(configuration['name'], configuration['mae']))


def get_features_for_configuration(configuration, df:pd.DataFrame):
    logging.info("Prepping for {}".format(configuration['name']))

    features = np.empty((len(df), 121, configuration['number_features']))
    all_features = np.empty((len(df), 121, 6))
    labels = np.empty((len(df), 37))
    doc_ids = np.empty((len(df)))
    for i, item in enumerate(df):
        doc_ids[i] = item['doc_id'][0]
        item = item.drop('doc_id', axis=1)
        all_features[i] = item.values[:601:5]
        labels[i] = item['cgmValue'].values[600::5]

    logging.info("feature shape {}".format(all_features.shape))
    features = all_features[:,:,configuration['columns']]
    logging.info("feature shape {}".format(features.shape))

    return features, labels, doc_ids

def train_model_for_configuration(x_train, x_test, y_train, y_test, configuration):
    model = None
    with tf.device('/GPU:0'):
            # define Model
            model = get_lstm_model(configuration['number_features'])
            #
            es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=100)
            # fit model
            history = model.fit(x_train, y_train, epochs = 2000, batch_size = 256 , validation_data = (x_test, y_test), callbacks=[es])
            test_acc = model.evaluate(x_test, y_test)
            model.save('{}models/1p-gpu-2000-1l-{}.h5'.format(path, configuration['name'])) 
            model.save_weights('{}models/w-1p-gpu-2000-1l-{}.h5'.format(path, configuration['name']))
            logging.info('Test mae: {}'.format(test_acc[2]))
            configuration['mae'] = test_acc[2]

    return model
def get_lstm_model(number_features):
    
    if tf.test.is_gpu_available():
        lstm_cell = CuDNNLSTM
    else:
        lstm_cell = LSTM
    
    model = Sequential()
    #model.add(lstm_cell(121, input_shape = (121, number_features), return_sequences= True)) 
    model.add(lstm_cell(121, input_shape = (121, number_features))) 
    #model.add(Dropout(0.5))
    #model.add(lstm_cell(40, return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(lstm_cell(40))
    model.add(Dense(37))
    model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy', 'mae'])
    model.summary()

    return model


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

        history = model.fit(x_train, y_train, epochs = 2000, batch_size = 256 , validation_data = (x_test, y_test))
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
    data_object['doc_id'] = di.doc_id
    data_object['data'] = (pd.DataFrame(json.loads(di['data'])))
    data_object['features-90'] = di['features-90']
    data_object['start_time'] = di['start_time']
    if 'result' in di:
        data_object['result'] = di['result']
    # for key in etypes:
    #    if key in di:
    #        df = pd.DataFrame(json.loads(di[key]))
    #        df.index = list(map(lambda x: float(x), df.index))
    #        data_object.__setattr__(key + '_events', df.sort_index())
    return data_object

def check_time_test(item):
    time = pd.Timestamp(item['start_time'])
    if time.day in [5,12]:
        return True
    if time.day in [4,11] and time.hour > 10:
        return True
    if time.day in [6,13] and time.hour < 14:
        return True
    return False

def load_data_with_result(db):
    logging.info("START execution")
    logging.info("Get all elements from DB")
    all_items = db.search(where('features-90').exists())
    #all_items = list(filter(lambda x: len(x['result']) == 5, all_items))

    #all_items = list(filter(lambda x: pd.Timestamp(x['start_time']).day not in [3,4,5,11,12,13], all_items)) # 1 Patient train filter
    #all_items = list(filter(check_time_test, all_items))
    #all_items = all_items[0:100]

    # filter elements by patient id, relevant for db with more than 1 patient
    #all_items = list(filter(lambda x: x['id'] == '82923830', all_items))  # TRAIN patient
    all_items = list(filter(lambda x: x['id'] == '27283995', all_items))  # Test patient

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
    df = df.sort_index()
    df['features-90'] = 0
    df['time_of_day'] = get_time_of_day(data_object['start_time'])
    for i, v in enumerate(range(0,600,15)):
        df.loc[v,'features-90'] = data_object['features-90'][i]
    df['doc_id'] = data_object['doc_id']
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
  

def get_time_of_day(start_time:str)->pd.Series:
    logging.debug("get Time of day")
    time = pd.Timestamp(start_time)
    time_of_day = pd.Series([0] * 780)
    for offset in range(780):
        logging.debug("offset: {}".format(offset))
        t = time + pd.Timedelta('{}M'.format(offset))
        logging.debug("time: {}".format(t))
        hour = t.hour
        logging.debug("hour: {}".format(hour))
        category = int((hour - 2) / 4)
        logging.debug("cat: {}".format(category))
        time_of_day[offset] = category

    return time_of_day

def shift_to_zero(df: pd.DataFrame)-> (pd.DataFrame, float):
    logging.debug("shift to zero")
    logging.debug("cgm at 600: {}".format(df['cgmValue'][600]))
    logging.debug("cgm at 100: {}".format(df['cgmValue'][100]))
    offset = df['cgmValue'][600]
    logging.debug("offset: {}".format(offset))

    df['cgmValue'] -= offset
    logging.debug("cgm at 600: {}".format(df['cgmValue'][600]))
    logging.debug("cgm at 100: {}".format(df['cgmValue'][100]))
    return df, offset


def plot_lstm_predictions(model, x : np.ndarray, y : np.ndarray):
    logging.info("Plot a few predictions")
    predictions = model.predict(x)


    for i in range(min(len(x), 20)):
        logging.info("i: {}".format(i))
        plt.plot(range(0,601,5), x[i,:,0],label='features')
        plt.plot(range(600,781,5), y[i], label='Real CGM')
        plt.plot(range(600,781,5), predictions[i], label='Prediction')
        plt.legend()
        plt.savefig("{}results/plots/lstm-6-{}".format(path, i))
        plt.close()

def predict_with_model(features, labels, doc_ids, model):
    logging.info("Making predictions!")
    predictions = model.predict(features)
    logging.info(len(predictions))
    logging.info(len(doc_ids))

    prediction_objects = list(map(lambda x: {'predictions': x[0], 'doc_id': x[1]}, zip(predictions, doc_ids)))

    return prediction_objects


def save_predictions(db:TinyDB, predictions: {}):
    for item in predictions:
        logging.info("item {}".format(item['doc_id']))
        db.update({'lstm-test-result': item['predictions'].tolist()}, doc_ids=[item['doc_id']])

    db.storage.flush()

if __name__ == "__main__":
    main()

