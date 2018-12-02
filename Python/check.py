import pandas

import extractor
from datetime import datetime

from Classes import UserData
from predict import calculateBG
from matplotlib import pyplot as plt

timeFormat = "%d.%m.%y,%H:%M"

def checkCurrent(data, udata):

    events = extractor.getEvents(data)

    converted = events.apply(lambda event: convertTimes(event, "08.03.18,18:00"))
    df = pandas.DataFrame([vars(e) for e in converted])

    df = df[df.time > 0]
    df = df[df.time < udata.simlength * 60]

    print(df)
    #df = df.iloc[[12]]


    dataZero = data[data['time'] == "08:58"]
    initialBG = dataZero['cgmValue'].values[0]

    udata.bginitial = initialBG

    data = calculateBG(df, udata, 100)

    basalValues = df[df.etype == 'tempbasal']
    carbValues = df[df.etype == 'carb']
    bolusValues = df[df.etype == 'bolus']


    for i in range(0,len(data[0])):
        print(str(data[3][i]) + '\t' * 5 + str(data[0][i]) + '\t' * 5 + str(data[1][i]) + '\t' * 5 + str(data[2][i]) )

    dataasdf = pandas.DataFrame([data[1], data[0], data[1], data[2]])
    print(dataasdf)

    plt.plot(data[3], data[0])
    plt.plot(data[3], data[1])
    plt.plot(data[3], data[2])

    plt.plot(basalValues.time, [0] * len(basalValues), "bo")
    plt.plot(carbValues.time, [0] * len(carbValues),"go")
    plt.plot(bolusValues.time, [0] * len(bolusValues),"ro")
    plt.savefig("result.png", dpi=600)


def convertTimes(event, start):
    startTime = datetime.strptime(start, timeFormat)
    eventTime = datetime.strptime(event.time, timeFormat)
    timeDifference = eventTime - startTime
    if eventTime < startTime:
        timeDifference = startTime - eventTime
        event.time = - timeDifference.seconds / 60
    else:
        event.time = timeDifference.seconds / 60
    if event.etype == "tempbasal":
        t1 = datetime.strptime(event.t1, timeFormat)
        event.t1 = event.time
        #timeDifference = eventTime - t1
        #event.t1 = timeDifference.seconds
        #t2 = datetime.strptime(event.t2, timeFormat)
        #timeDifference = eventTime - t2
        event.t2 = event.t1 + 2 * 60 # TODO better estimate
    #print(event.time, event.t1, event.t2)
    return event