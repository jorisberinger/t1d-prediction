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


def check_data_object(data_object: DataObject) -> bool:
    valid_events: bool = check_events(data_object)
    valid_cgm: bool = check_cgm(data_object)
    valid: bool = valid_events and valid_cgm
    if not valid:
        return valid

    return valid


def check_events(data_object:DataObject) -> bool:
    # check for events in the prediction time frame
    event_types = ['carb', 'bolus'] # basal events are fine
    for key in event_types:
        if hasattr(data_object, key + '_events'):
            events = data_object.__getattribute__(key + '_events')
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
    cgm_values = data_object.data['cgmValue_original'].dropna()
    index = cgm_values.index.values 
    if len(index) < 10:
        logging.debug("index less than 10 items")
        return False
    differences = index[1:-1] - index[0:-2]
    return max(differences) < 75







