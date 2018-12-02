import json

from readData import read_data
import extractor
import uuid
from datetime import datetime
import pandas
timeFormat = "%d.%m.%y,%H:%M%z"
filename = "../../Data/data0318.csv"
path = "../Autotune/data/input/"

"""
{
    "_id": "5bff56bedb927d5b737433af",
    "type": "sgv",
    "date": 1543460400000,
    "dateString": "2018-11-29T04:00:00+01:00",
    "device": "openaps://tmpaps",
    "sgv": 112,
    "direction": "Flat"
  },
  """
def createGlucoseJson(data):
    i = 0
    jsonarray = data.apply(lambda x: {"_id": uuid.uuid4().hex,
                                      "type" : "sgv",
                                      "dateString" : datetime.strptime(x.date + ',' + x.time + '+0100', timeFormat).isoformat(),
                                      "date" : datetime.strptime(x.date + ',' + x.time + '+0100', timeFormat).timestamp() * 1000,
                                      "sgv" : x.cgmValue,
                                      "device": "openaps://tmpaps"}, axis=1)

    file = open(path + 'glucose.json', 'w')
    file.write(json.dumps(jsonarray.values.tolist()))

def main():
    # read data and create json files for autotune
    data = read_data(filename)
    data = data[data['date'] == "08.03.18"]

    cgms = extractor.getBGContinious(data)

    createGlucoseJson(cgms)
if __name__ == '__main__':
    main()