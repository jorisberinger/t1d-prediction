import json

import extractor
import uuid
from datetime import datetime
import pandas
import logging
import os

logger = logging.getLogger(__name__)


timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"

path = os.getenv('T1DPATH', '../')
folder_path = path + "data/input/1/"

# Convert Time from String to Datetime object
def convertTime(dateandtime):
    return datetime.strptime(dateandtime + timeZone, timeFormat)


# write Json data in file. Filename is appended to path variable to get relative path of file
def writeJson(jsonarray, filename):
    file = open(folder_path + filename + '.json', 'w')
    file.write(json.dumps(jsonarray))


# create Glucose Json in order to use it as input for autotune
def createGlucoseJson(data):
    i = 0
    grouped = data.groupby(['date'])
    for name, group in grouped:
        logger.debug("glucose - " + name)
        jsonarray = group.apply(lambda x: {"_id": uuid.uuid4().hex,
                                                                 "type": "sgv",
                                                                 "dateString": convertTime(
                                                                     x.date + ',' + x.time).isoformat(),
                                                                 "date": convertTime(
                                                                     x.date + ',' + x.time).timestamp() * 1000,
                                                                 "sgv": x.cgmValue,
                                                                 "device": "openaps://tmpaps"}, axis=1)
        writeJson(jsonarray.values.tolist(), "glucose-" + name)





def bolusToJson(event):
    return {"eventType": "Correction Bolus",
            "duration": 0,
            "insulin": event.units,
            "timestamp": convertTime(event.time).isoformat(),
            "_id": uuid.uuid4().hex,
               "bolus": {
                   "timestamp": convertTime(event.time).isoformat(),
                   "_type": "Bolus",
                   "amount": event.units,
                   "programmed": event.units,
                   "unabsorbed": 0,
                   "duration": 0
               },
            "created_at": convertTime(event.time).isoformat(),
            "notes": "Normal bolus (solo, no bolus wizard).\nCalculated IOB: 0.279\nProgrammed bolus 0.4\nDelivered bolus 0.4\nPercent delivered:  100%",
            "medtronic": "mm://openaps/mm-format-ns-treatments/Correction Bolus",
            "enteredBy": "openaps://medtronic/754",
            "carbs": None
            }


def basalToJson(event):
    return {"eventType": "Temp Basal",
            "duration": 30,
            "rate": event.dbdt,
            "absolute": event.dbdt,
            "timestamp": convertTime(event.time).isoformat(),
            "_id": uuid.uuid4().hex,
            "raw_rate": {
                "timestamp": convertTime(event.time).isoformat(),
                "_type": "TempBasal",
                "temp": "absolute",
                "rate": event.dbdt
            },
            "raw_duration": {
                "timestamp": convertTime(event.time).isoformat(),
                "_type": "TempBasalDuration",
                "duration (min)": 30
            },
            "medtronic": "mm://openaps/mm-format-ns-treatments/Temp Basal",
            "created_at": convertTime(event.time).isoformat(),
            "enteredBy": "openaps://medtronic/754",
            "carbs": None,
            "insulin": None
            }



def eventToJson(e):
    event = e.values.tolist()[0]
    if event.etype == "bolus":
        return bolusToJson(event)
    elif event.etype == "tempbasal":
        return basalToJson(event)


# create insulin pump history Json in order to use it as input for autotune
def createPumphistoryJson(events):
    data = pandas.DataFrame(events)
    data['date'] = data.apply(lambda x:
                              x.values[0].time.split(',')[0], axis=1)
    grouped = data.groupby('date')
    for name, group in grouped:
        logger.debug("pumphistory - " + name)
        jsondata = group.apply(eventToJson, axis=1)
        jsondata = jsondata[jsondata.notnull()]
        writeJson(jsondata.values.tolist(), "pumphistory-"+name)


# read data and create json files for autotune
def prep_for_autotune(data):
    logger.info("prep for autotune - start")
    directory = os.path.dirname(folder_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create glucose.json from cgmValue
    cgms = extractor.getBGContinious(data)
    createGlucoseJson(cgms)

    # create pumphistory.json from insulin events
    events = extractor.getEvents(data)
    createPumphistoryJson(events)
    logger.info("prep for autotune - end")
