import logging
from tinydb import TinyDB, where
import pandas as pd
import random
from data.checkData import check_data_object
from data.convertData import convert, interpolate_cgm
from data.dataObject import DataObject
from data.dataPrep import add_timestamps, add_to_db, prepare_data
from data.readData import read_data


# Class for loading data into the Tiny DB
class DataLoader:
    db: TinyDB

    # Init Data Loader
    def __init__(self, db: TinyDB):
        # set TinyDB instance to work with
        self.db = db

    # Read data from file and put into database
    def load(self, filename):
        # Read file as pandas Dataframe
        data_original = read_data(filename)
        # Prepare Data
        data = prepare_data(data_original)
        # add data to tiny db
        count = add_to_db(data, self.db)
        logging.info("added {} new items to DB".format(count))

    def add_events(self):
        count = 0
        # Iterate through all items in db
        for item in self.db.search(~where('carb').exists() | ~where('bolus').exists() | ~where('basal').exists()):
            if not hasattr(item, 'carb'):
                data_object = DataObject.from_dict(item)
                logging.debug("doc id: {}".format(item.doc_id))
                data_object.add_events()
                di = data_object.to_dict()
                self.db.update(di, where('start_time') == di['start_time'])
                count += 1
        self.db.storage.flush()
        logging.info("updated {} items in DB".format(count))

    def check_valid(self):
        # get all items which are not flagged valid
        items = self.db.search(~where('valid').exists())
        items = self.db.all()
        random.shuffle(items)
        logging.info("checking {} items".format(len(items)))
        remove_ids = []
        valid_ids = []
        valid_events = []
        valid_cgms = []
        issues = []
        for item in items:
            valid, v_events, v_cgm, issue = check_data_object(DataObject.from_dict(item))
            valid_events.append(v_events)
            valid_cgms.append(v_cgm)
            if not v_cgm:
                issues.append(issue)
            if valid:
                valid_ids.append(item.doc_id)
            if not valid:
                remove_ids.append(item.doc_id)
        # remove invalid items
        self.db.remove(doc_ids = remove_ids)
        # remember valid items
        logging.info("not valid events {}".format(len(valid_events) - sum(valid_events)))
        logging.info("not valid cgms {}".format(len(valid_cgms) - sum(valid_cgms)))
        logging.info("less than 10: {}".format(len(issues) - sum(issues)))
        logging.info("max difference < 75: {}".format(sum(issues)))
        self.db.update({'valid': True}, doc_ids=valid_ids)
        self.db.storage.flush()
        logging.info("removed {} items form db".format(len(remove_ids)))
        logging.info("Still {} items in db".format(len(self.db)))
