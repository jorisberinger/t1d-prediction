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
from keras.layers import CuDNNLSTM, Dropout, Dense, LSTM, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
import keras
coloredlogs.install(level = logging.INFO, fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')

#db_path = path + 'data/tinydb/dbtest2.json'
# db_path = path + 'data/tinydb/db4p.json'
db_path = path + 'data/tinydb/dbtest2.json'

# patient = 'mp'
# patient = '3p'
patient = '1p'

# model_path = path+'models/'+patient+ '-err-gpu-2000-1l-cgm, insulin, carbs, optimized, tod.h5'
#model_path = "{}models/{}-error-best-model".format(path,patient)
model_path = path+'models/3p-arima-error-100-1l-cgm, insulin, carbs, optimized, tod.h5'

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
    logging.info("Start patching...")
    # Load database
    db = TinyDB(db_path, storage = CachingMiddleware(JSONStorage))
    logging.info("database loaded with {} entries".format(len(db)))
    # convert db items into features and labels
    features, labels, doc_ids = load_data_with_result(db, cache=False)

    #features, labels, doc_ids = features[:100], labels[:100], doc_ids[:100]
    features = normalize(features)
    labels = normalize_labels(labels)
    # Split data into train and testing data
    #x_train, x_test, y_train, y_test, doc_ids_train, doc_ids_test = train_test_split(features, labels, doc_ids, test_size = 0.2, shuffle = True, random_state=1)
    x_train, x_test, y_train, y_test, doc_ids_train, doc_ids_test = train_test_split_custom(features, labels, doc_ids, db)

    # learn error model
    model = learn_errors(x_train, x_test, y_train, y_test, db, cache= False)
    
    # predict error for predictor
    predicted_errors = predict_with_model(x_test, y_test, doc_ids_test, model)

    # get new prediction
    new_predictions = list(map(lambda x: get_new_prediction(x, db=db), predicted_errors))
    #exit()
    # save the new predictions
    success = list(map(lambda x: save_prediction(db, x), new_predictions))
    #success = list(map(lambda x: save_predictions_lstm(db, x), predicted_errors))

    db.storage.flush()
    logging.info("Updated all succesfully: {}".format(all(success)))
    logging.info("Done")




def learn_errors(x_train, x_test, y_train, y_test , db: TinyDB, cache: bool):

    if cache:
        model = keras.models.load_model(model_path)
        model.summary()
        return model

    # train model with training data
    # configuration = configurations[0]
    models = []
    for configuration in configurations:
        configuration['number_features'] = len(configuration['columns'])
        model = train_model_for_configuration(x_train, x_test, y_train, y_test, configuration)
        models.append(model)
    
    for configuration in configurations:
        logging.info("{} with mae {}".format(configuration['name'], configuration['mae']))

    return models[0]


def get_new_prediction(item, db):
    logging.debug(item)
    db_item = db.get(doc_id=item['doc_id'])
    #lstm_results = list(filter(lambda x: 'LSTM' in x['predictor'] and '5000' not in x['predictor'], db_item['result']))[0]['errors']
    # old_prediction = list(filter(lambda x: 'Arima' in x['predictor'], db_item['result']))[0]['errors']
    old_prediction = list(filter(lambda x: 'Optimizer' in x['predictor'] and '15' not in x['predictor'], db_item['result']))[0]['errors']
    predicted_error = item['predictions']
    logging.debug(old_prediction)
    logging.debug(predicted_error)
    new_prediction = old_prediction - predicted_error
    old_err = sum(map(abs,old_prediction))
    new_err = sum(map(abs,new_prediction))
    logging.debug("new Prediction error:\n{}".format(new_prediction))
    logging.debug("old: {}\tnew: {}".format(old_err, new_err))
    logging.info("{} - {}".format(old_err - new_err, (old_err - new_err) < 0))

    return {'predictions': new_prediction, "doc_id": item['doc_id']}



def get_features_for_configuration(configuration, df:pd.DataFrame):
    logging.info("Prepping for {}".format(configuration['name']))

    features = np.empty((len(df), 121, configuration['number_features']))
    all_features = np.empty((len(df), 121, 7))
    labels = np.empty((len(df), 8))
    doc_ids = np.empty((len(df)))
    for i, item in enumerate(df):
        doc_ids[i] = item['doc_id'][0]
        item = item.drop('doc_id', axis=1)
        all_features[i] = item.values[:601:5]
        labels[i] = item['optimizer_error'].values[:8]
        # labels[i] = item['arima_error'].values[:8]
        #labels[i] = item['cgmValue'].values[600::15]


    logging.info("feature shape {}".format(all_features.shape))
    features = all_features[:,:,configuration['columns']]
    logging.info("feature shape {}".format(features.shape))

    return features, labels, doc_ids

def train_model_for_configuration(x_train, x_test, y_train, y_test, configuration):
    logging.info("x train shape {}".format(x_train.shape))
    logging.info("x test shape {}".format(x_test.shape))
    x_train, x_test = x_train[:,:,configuration['columns']], x_test[:,:,configuration['columns']]
    model = None
    with tf.device('/GPU:0'):
            # define Model
            model = get_lstm_model(configuration['number_features'])
            # model = get_nn_model(configuration['number_features'])
            #
            #es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=100)
            # fit model
            #history = model.fit(x_train, y_train, epochs = 2000, batch_size = 256 , validation_data = (x_test, y_test), callbacks=[es])
            checkpointer = ModelCheckpoint(filepath="{}models/{}-opti-best-model".format(path,patient), monitor='val_acc',verbose=1, save_best_only=True)
            logging.info("Y TRAIN")
            logging.info(y_train[0:20])
            history = model.fit(x_train, y_train, epochs = 1000, batch_size = 128 , validation_data = (x_test, y_test), callbacks=[checkpointer])
            test_acc = model.evaluate(x_test, y_test)
            model.save('{}models/{}-opti-error-100-1l-{}.h5'.format(path,patient, configuration['name'])) 
            model.save_weights('{}models/w-{}-opti-error-100-1l-{}.h5'.format(path, patient, configuration['name']))
            logging.info('Test mae: {}'.format(test_acc[2]))
            configuration['mae'] = test_acc[2]

    return model


def get_nn_model(number_features):
    model = Sequential()
    model.add(Dense(50, input_shape=(121,number_features), activation='relu'))
    model.add(Dense(50, activation='relu'))
    # model.add(Dense(120))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.compile(optimizer= 'adam', loss='mse', metrics=['accuracy', 'mae'])
    model.summary()
    return model

def get_lstm_model(number_features):
    
    if tf.test.is_gpu_available():
        lstm_cell = CuDNNLSTM
    else:
        lstm_cell = LSTM
    
    model = Sequential()
    #model.add(lstm_cell(121, input_shape = (121, number_features), return_sequences= True)) 
    #model.add(Dropout(0.2))
    model.add(lstm_cell(20, input_shape = (121, number_features))) 
    #model.add(Dropout(0.5))
    #model.add(lstm_cell(40, return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(lstm_cell(40))
    # model.add(Dense(30))
    model.add(Dense(8, activation='relu'))
    model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy', 'mae'])
    model.summary()

    return model


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

def check_time_test(time):
    if time.day in [5,12]:
        return True
    if time.day in [4,11] and time.hour > 10:
        return True
    if time.day in [6,13] and time.hour < 14:
        return True
    return False

def check_time_train(time):
    return time.day not in [4,5,6,11,12,13]

def load_data_with_result(db: TinyDB, cache: bool):
    logging.info("Start getting data..")
    if cache:
        features = np.fromfile(path+'models/npe-arima-error-'+patient+ '-all-features')
        labels = np.fromfile(path+'models/npe-arima-error-'+patient+ '-all-labels')
        doc_ids = np.fromfile(path+'models/npe-arima-error-'+patient+ '-all-docids')
        nr_items = len(doc_ids)
        features = features.reshape((nr_items, 121, 6))
        labels = labels.reshape((nr_items, 8))
        doc_ids = doc_ids.reshape((nr_items,))

    else:


        logging.info("Get all elements from DB")
        all_items = db.search((where('valid') == True) & where('features-90').exists() )
        #all_items = list(filter(lambda x: len(x['result']) == 5, all_items))
        #all_items = list(filter(lambda x: pd.Timestamp(x['start_time']).day not in [3,4,5,11,12,13], all_items)) # 1 Patient train filter
        #all_items = list(filter(check_time_test, all_items))
        # all_items = list(filter(check_time_train, all_items))
        # filter elements by patient id, relevant for db with more than 1 patient
        if patient == '3p':
            all_items = list(filter(lambda x: x['id'] == '82923830', all_items))  # TRAIN patient
        elif patient == '2p':
            all_items = list(filter(lambda x: x['id'] == '27283995', all_items))  # Test patient
        elif patient == '1p':
            pass
        elif patient == 'mp':
            pass
        else:
            raise Exception("Unknown patient")

        #all_items = list(filter(lambda x: len(x['result']) >=5, all_items))

        #all_items = all_items[0:100]
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

        # feature configuration
        configuration = configurations[0]
        configuration['number_features'] = len(configuration['columns'])
        # Get features and labels
        features, labels, doc_ids = get_features_for_configuration(configuration=configuration, df=subset)
        # # save features, labels and doc id
        features.tofile(path+'models/npe-arima-error-'+patient+ '-all-features')
        labels.tofile(path+'models/npe-arima-error-'+patient+ '-all-labels')
        doc_ids.tofile(path+'models/npe-arima-error-'+patient+ '-all-docids')

    return features, labels, doc_ids 

def get_feature_list(data_object):
    df = data_object['data'][['cgmValue', 'basalValue', 'bolusValue', 'mealValue']].fillna(0)
    df.index = list(map(float , df.index))
    df = df.sort_index()
    df['features-90'] = 0
    df['time_of_day'] = get_time_of_day(data_object['start_time'])
    for i, v in enumerate(range(0,600,15)):
        df.loc[v,'features-90'] = data_object['features-90'][i]
    df['doc_id'] = data_object['doc_id']
    # df['arima_error'] = pd.Series(list(filter(lambda x: 'Arima' in x['predictor'], data_object['result']))[0]['errors'])
    df['optimizer_error'] = pd.Series(list(filter(lambda x: 'Optimizer' in x['predictor'] and '15' not in x['predictor'], data_object['result']))[0]['errors'])
    #df['lstm_error'] = pd.Series(list(filter(lambda x: 'LSTM' in x['predictor'] and '1000' not in x['predictor'], data_object['result']))[0]['errors'])

    return df


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
    predictions = model.predict(features, verbose=1)
    logging.info(len(predictions))
    logging.info(len(doc_ids))
    logging.info("PREDICTIONS - before")
    logging.info(predictions[0:10])
    predictions = predictions * 300
    # logging.info("PREDICTIONS - after")
    # logging.info(predictions[0:10])
    prediction_objects = list(map(lambda x: {'predictions': x[0], 'doc_id': x[1]}, zip(predictions, doc_ids)))

    return prediction_objects


# def save_predictions(db:TinyDB, predictions: {}):
#     for item in predictions:
#         logging.info("item {}".format(item['doc_id']))
#         db.update({'err-test-1-result': item['predictions'].tolist()}, doc_ids=[item['doc_id']])

#     db.storage.flush()

def save_prediction(db:TinyDB, item: {}):
    logging.info("item {}".format(item['doc_id']))
    db.update({'optimizer-error-prediction-1000': item['predictions'].tolist()}, doc_ids=[item['doc_id']])
    return True

def save_predictions_lstm(db:TinyDB, item: {}):
    logging.info("item {}".format(item['doc_id']))
    db.update({'lstm-result-1000': item['predictions'].tolist()}, doc_ids=[item['doc_id']])
    return True


def train_test_split_custom(features, labels, doc_ids, db):
    logging.info("Split data into training and test")
    x_train, x_test, y_train, y_test, doc_ids_train, doc_ids_test = [], [], [], [], [], [] 
    # p2_doc_ids = list(map(lambda x: x.doc_id, db.search((where('id') == '82923830') & (where('valid') == True))))
    # p3_doc_ids = list(map(lambda x: x.doc_id, db.search((where('id') == '27283995') & (where('valid') == True))))
    # p4_doc_ids = list(map(lambda x: x.doc_id, db.search((where('id') == '29032313') & (where('valid') == True))))
    all_items = db.search(where('valid') == True)
    for feature, label, doc_id in zip(features, labels, doc_ids):
        
        start_time = pd.Timestamp(list(filter(lambda x: x.doc_id == doc_id, all_items))[0]['start_time'])
        
        if check_time_train(start_time):
            x_train.append(feature), y_train.append(label), doc_ids_train.append(doc_id)
        if check_time_test(start_time):
            x_test.append(feature), y_test.append(label), doc_ids_test.append(doc_id)
        # else:
        #     raise Exception("Doc id not found")
    logging.info("training length: {}\ttest length: {}".format(len(x_train), len(x_test)))
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), np.array(doc_ids_train), np.array(doc_ids_test)


def normalize(features):
    logging.info("Normalizing features")

    max_values = features.max(axis=1).max(axis=0)
    max_values[0] = 1
    logging.info(max_values)
    for i, row in enumerate(features):
        features[i] = row / max_values
    
    max_values = features.max(axis=1).max(axis=0)
    logging.info(max_values)

    return features

def normalize_labels(labels):
    logging.info("Normalizing labels")
    normalizing_factor = 1 / 300
    labels = labels * normalizing_factor

    return labels




if __name__ == "__main__":
    main()

