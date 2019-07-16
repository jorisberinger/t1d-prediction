import logging

import coloredlogs

import re
import os
import gzip
import shutil
import json
import pandas as pd
from matplotlib import pyplot as plt
from tinydb import TinyDB, JSONStorage, where
from tinydb.middlewares import CachingMiddleware
import main_Prep as prep


coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')


path = 'c:\\users\\joris\\desktop\\data'

db_path = os.getenv('T1DPATH', '../../') + 'data/tinydb/db_multi.json'


def main():
    logging.info("starting json converter")
    csv_to_db()
    #json_to_csv()


def csv_to_db():
    logging.info("CSV to DB")
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))
    logging.info("number of csv: {}".format(len(files)))

    db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
    for file in files:
        logging.info("filename: {}".format(file))
        prep.main(db, file)
        break

    logging.info("done")



def csv_cleanup():
    logging.info("clean up csv")
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))
    logging.info("number of csv: {}".format(len(files)))


    lengths = []
    for file in files[5:10]:
        df = pd.read_csv(file, low_memory=False)
        if len(df) > 10000:
            lengths.append(len(df))
            logging.info("length of df: {}".format(len(df)))
            df = df.drop(
                [   '_id', '_tell', 'body', 'data_type', 'device', 'display_time', 'name', 'Unnamed: 0', 'date_type',
                   'packet_size', 'raw', 'trend_arrow', 'type', '_body',
                   '_date', '_description', '_head', '_id.1', '_type',
                   'alarm_description', 'alarm_type',
                   'changed', 'created_at', 'enteredBy', 'eventType', 'fixed',
                   'medtronic', 'notes',
                    'raw_duration', 'raw_rate', 'reason', 'stale', 'tail',
                   'targetBottom', 'targetTop', 'temp', 'timestamp', 'type.1',
                   'wizard']
            ,axis=1, errors='ignore')
            logging.info("columns: {}".format(df.columns))


    plt.boxplot(lengths)
    plt.show()


def json_to_csv():
    logging.info("converting json files to csv")
    files = []
    groups =[]
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for file in files:
        if 'entries' in file:
            logging.info(file)
            regex = re.compile(file.replace('\\', '\\\\').replace('entries', '(devicestatus|entries|profile|treatments)'))
            #logging.warning(regex)

            group = list(filter(lambda x: regex.match(x), files))
            #for g in group:
                #logging.error(g)
            groups.append(group)

    for group in groups:
        try:
            csv = import_json(group)
        except Exception:
            logging.error("got exception for {}".format(group[0]))


def import_json(filenames):
    logging.info("import json")
    data_frame : pd.DataFrame = pd.DataFrame()
    for filename in filenames:
        logging.info(filename)
        if 'entries' in filename:
            logging.info("Getting svg from Entries")
            df = pd.read_json(filename, precise_float = True)
            sgv = df['sgv']
            sgv.index = df['date']
            sgv.name ="cgmValue"
            data_frame['cgmValue'] = sgv
            #data_frame = pd.concat([data_frame, sgv], axis = 1)
        elif 'treatments' in filename:
            df = pd.read_json(filename, precise_float = True)
            df.index = df['timestamp']
            logging.info("Getting Insulin data from treatments")
            carbs = df[['carbs','absorptionTime']]
            carbs.name = 'carbs'
            carbs = carbs[carbs['carbs'].notna()]
            carbs = carbs[~carbs.index.duplicated()]
            data_frame = pd.concat([data_frame, pd.DataFrame(carbs)], sort=True)
            logging.info("carbs inserted")
            temp_basal = df[df['eventType'] == 'Temp Basal']['absolute']
            temp_basal.name = 'temp_basal'
            temp_basal = temp_basal[temp_basal.notna()]
            temp_basal = temp_basal[~temp_basal.index.duplicated()]
            data_frame = pd.concat([data_frame, pd.DataFrame(temp_basal)], sort = True)
            logging.info("added temp basal")
            bolus = df['insulin']
            bolus.name = 'bolus'
            bolus = bolus[bolus.notna()]
            bolus = bolus[~bolus.index.duplicated()]
            data_frame = pd.concat([data_frame, pd.DataFrame(bolus)], sort = True)
            logging.info("added correction bolus")
            #data_frame = pd.concat([data_frame, df], axis = 1)
    data_frame = data_frame.sort_index()
    csv_filename = list(filter(lambda x: 'entries' in x, filenames))[0].replace('entries','csv').replace('json','csv')
    data_frame.to_csv(csv_filename)
    logging.info("done")




def remove_gz_files():
    logging.info("remove gz files")
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        if '.gz' in f:
            logging.info("removing {}".format(f))
            os.remove(f)


def unzip_all():
    logging.info("unzip all json files")
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        if '.gz' in f:
            logging.info(f)
            filename = f[:-3]
            logging.info(filename)
            with gzip.open(f, 'rb') as f_in:
                with open(filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    main()