import logging
import os
from datetime import timedelta

import coloredlogs
import numpy as np
import pandas as pd
from tinydb import Query, TinyDB, where
from tinydb.middlewares import CachingMiddleware

from data.convertData import (convert, create_time_index, interpolate_cgm,
                              select_columns, copy_cgm)
from data.dataConnector import DataConnector
from data.dataObject import DataObject
from data.readData import read_data

path = os.getenv('T1DPATH', '../../')

window_length = 13
long_window_length = 63
delta_length = 15



def prepare_data(data_original: pd.DataFrame) -> pd.DataFrame:
    # convert index to time index
    data = convert(data_original)
    # Copy CgmValue column to save original values
    data = copy_cgm(data)
    # interpolate Data to minute cgm resolution
    data = interpolate_cgm(data)
    # add timestamp as int(to recover date if lost when converting to json)
    data = add_timestamps(data)

    return data


def add_to_db(data: pd.DataFrame, db: TinyDB) -> [pd.DataFrame]:
    # split data in to prediction windows
    # set start and end time
    start_time = data.index[0] +timedelta(hours = long_window_length - window_length)
    final_start_time = data.index[-1] - timedelta(hours = window_length)
    counter = 0
    # get all starting times
    times = list(map(lambda x: x['start_time'], db.all()))
    # Loop Through Data
    while start_time < final_start_time :
        # Check if time frame exists
        if not start_time.isoformat() in times:
            data_object = DataObject()
            data_object.set_start_time(start_time)
            end_time = start_time + timedelta(hours = window_length)
            data_object.set_end_time(end_time)
            # select data for this window --- SHORT
            subset_short = data.loc[start_time <= data.index]
            subset_short = subset_short.loc[end_time >= subset_short.index]
            # set index to minutes
            subset_short.index = np.arange(0.0, len(subset_short))
            data_object.set_data_short(subset_short)
             # select data for this window --- LONG
            subset_long = data.loc[start_time - timedelta(hours= long_window_length - window_length) <= data.index]
            subset_long = subset_long.loc[end_time >= subset_long.index]
            # set index to minutes
            subset_long.index = np.arange(0.0, len(subset_long))
            data_object.set_data_long(subset_long)
            db.insert(data_object.to_dict())
            counter += 1
            #if counter >= 1:
            #    break
            print('Processing [%d]\r'%counter, end="")
        start_time += timedelta(minutes = delta_length)
    db.storage.flush()
    return counter


def add_gradient(db: TinyDB):
    logging.debug("Calculate Gradient for items")
    gradients = []
    items = db.search(~where('gradient-s').exists())
    for item in items:
        logging.debug("#: {}\tdoc id: {}".format(len(gradients), item.doc_id))
        data_object = DataObject.from_dict(item)
        data = data_object.data['cgmValue'].values
        d1 = data[630:690]
        d2 = data[635:695]
        max_gradient = max(d2-d1)
        logging.debug("max gradient {}".format(max_gradient))
        item['gradient-s'] = max_gradient
        gradients.append(max_gradient)
        db.write_back([item])
    db.storage.flush()
    logging.info("Added gradient to {} items".format(len(items)))




def add_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    data['timestamp'] = pd.to_numeric(data.index)
    return data
