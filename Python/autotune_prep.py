import json

from readData import read_data
import extractor
import uuid
from datetime import datetime
import pandas


timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"
datafilename = "../../Data/data0318.csv"
path = "../Autotune/data/input/"


# Convert Time from String to Datetime object
def convertTime(dateandtime):
    return datetime.strptime(dateandtime + timeZone, timeFormat)


# write Json data in file. Filename is appended to path variable to get relative path of file
def writeJson(jsonarray, filename):
    file = open(path + filename + '.json', 'w')
    file.write(json.dumps(jsonarray))


# create Glucose Json in order to use it as input for autotune
def createGlucoseJson(data):
    i = 0
    jsonarray = data.apply(lambda x: {"_id": uuid.uuid4().hex,
                                      "type" : "sgv",
                                      "dateString" : convertTime(x.date + ',' + x.time).isoformat(),
                                      "date" : convertTime(x.date + ',' + x.time).timestamp() * 1000,
                                      "sgv" : x.cgmValue,
                                      "device": "openaps://tmpaps"}, axis=1)
    writeJson(jsonarray.values.tolist(), "glucose")


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
    jsondata = data.apply(eventToJson, axis=1)
    jsondata = jsondata[jsondata.notnull()]
    writeJson(jsondata.values.tolist(), "pumphistory")


# read data and create json files for autotune
def main():
    data = read_data(datafilename)
    data = data[data['date'] == "08.03.18"]

    # create glucose.json from cgmValue
    cgms = extractor.getBGContinious(data)
    createGlucoseJson(cgms)

    # create pumphistory.json from insulin events
    events = extractor.getEvents(data)
    createPumphistoryJson(events)


if __name__ == '__main__':
    main()