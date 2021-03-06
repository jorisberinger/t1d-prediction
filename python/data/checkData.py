import pandas as pd
import logging
import numpy as np

from Classes import UserData
from data.dataObject import DataObject

logger = logging.getLogger(__name__)


def check_window(window: pd.DataFrame, user_data: UserData) -> bool:
    # check that there is data at every check point
    t = np.arange(0., user_data.simlength * 60 - user_data.predictionlength + 1, 15.)
    t = np.append(t, user_data.simlength * 60.)
    selected = window.loc[t]
    nas = selected['cgmValue'].isna()
    return not nas.any()


def check_data_object(data_object: DataObject, early: bool) -> bool:
    if early:
        valid_cgm, issue = check_cgm(data_object)
        return valid_cgm, None, None, None
    else:
        valid_events: bool = check_events(data_object)
        valid_cgm, issue = check_cgm(data_object)
        valid: bool = valid_events and valid_cgm
        return valid, valid_events, valid_cgm, issue


def check_events(data_object:DataObject) -> bool:
    # check for events in the prediction time frame
    event_types = ['carb', 'bolus'] # basal events are fine
    # check that there is at least a carb or a bolus event
    if not any(map(lambda x: hasattr(data_object, x + '_events'), event_types)):
        return False
    for key in event_types:
        if hasattr(data_object, key + '_events'):
            events = data_object.__getattribute__(key + '_events')
            if key == 'bolus':
                events = events[events['units'] > 0]
            # Check that there are no events in the prediction window
            if not events.index[np.logical_and(events.index >= 600, events.index < 730)].empty:
                logging.debug("events in prediction ")
                return False
            # Check that there are events in the training window
            if events.index[events.index <= 600].empty:
                return False
    return True

def check_cgm(data_object: DataObject) -> bool:
    # Check that there are no gaps in cgm_values
    cgm_values = data_object.data_long['cgmValue_original'].dropna()
    index = cgm_values.index.values
    # Check that the first and last value are close to the start and end
    if len(index) < 60:
        logging.debug("index less than 60 items")
        return False, 0
    if index[0] > 90 or index[-1] < 3570:
        return False, 1
    differences = index[1:-1] - index[0:-2]
    if max(differences) >= 180:
        return False, 1
    
    cgm_values = data_object.data['cgmValue_original'].dropna()
    index = cgm_values.index.values

    if len(index) < 10:
        logging.debug("index less than 60 items")
        return False, 0
    if index[0] > 30 or index[-1] < 730:
        return False, 1
    differences = index[1:-1] - index[0:-2]
    if max(differences) >= 30:
        return False, 1

    return True, 0



def check_cgm_short(data_object: DataObject) -> bool:
    # Check that there are no gaps in cgm_values
    cgm_values = data_object.data['cgmValue_original'].dropna()
    index = cgm_values.index.values
    # Check that the first and last value are close to the start and end
    if len(index) < 10:
        logging.debug("index less than 10 items")
        return False, 0
    if index[0] > 30 or index[-1] < 750:
        return False, 1
    differences = index[1:-1] - index[0:-2]
    if max(differences) >= 75:
        return False, 1
    else:
        return True, 0







