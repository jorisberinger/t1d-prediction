import copy
import datetime
import json
import logging

import pandas as pd

from extractor import getEvents


etypes = ['carb', 'bolus', 'basal']
class DataObject:
    start_time: pd._libs.tslibs.timestamps.Timestamp
    end_time: pd._libs.tslibs.timestamps.Timestamp
    data: pd.DataFrame
    data_long: pd.DataFrame
    bolus_events: pd.DataFrame
    basal_events: pd.DataFrame
    carb_events: pd.DataFrame
    result: list

    def __init__(self):
        pass


    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def set_data_short(self, subset):
        subset.index = list(map(lambda x: float(x), subset.index))
        self.data = subset.sort_index()
    def set_data_long(self, subset):
        subset.index = list(map(lambda x: float(x), subset.index))
        self.data_long = subset.sort_index()

    def set_events(self, events):
        self.events = events

    def set_result(self, result):
        self.result = result

    def to_dict(self):
        cp = copy.deepcopy(self)
        cp.start_time = cp.start_time.isoformat()
        cp.end_time = cp.end_time.isoformat()
        cp.data = cp.data.to_json()
        cp.data_long = cp.data_long.to_json()
        for key in ['carb', 'bolus', 'basal']:
            if hasattr(self, key):
                cp.__setattr__(key, cp.__getattribute__(key).to_json())
        return cp.__dict__

    @classmethod
    def from_dict(cls, di):
        data_object = DataObject()
        data_object.set_start_time(pd.datetime.fromisoformat(di['start_time']))
        data_object.set_end_time(pd.datetime.fromisoformat(di['end_time']))
        data_object.set_data_short(pd.DataFrame(json.loads(di['data'])))
        data_object.set_data_long(pd.DataFrame(json.loads(di['data_long'])))
        if 'result' in di:
            data_object.set_result(di['result'])
        for key in etypes:
            if key in di:
                df = pd.DataFrame(json.loads(di[key]))
                df.index = list(map(lambda x: float(x), df.index))
                data_object.__setattr__(key + '_events', df.sort_index())
        return data_object

    def add_events(self):
        logging.debug("adding events")
        events = getEvents(self.data)
        events = pd.DataFrame([vars(e) for e in events], index=events.index)
        for key in etypes:
            self.__setattr__(key, events[events['etype'] == key])
        logging.debug("done adding events")

    def set_int_index(self):
        logging.debug("set int index")
        self.data.index = range(len(self.data))
