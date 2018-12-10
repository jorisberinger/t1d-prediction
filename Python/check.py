import pandas

import extractor
from datetime import datetime

from Classes import UserData
from predict import calculateBG
from matplotlib import pyplot as plt

timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"

def checkWindow(data, udata, startTime):
    events = extractor.getEvents(data)
    converted = events.apply(lambda event: convertTimes(event, startTime))
    df = pandas.DataFrame([vars(e) for e in converted])
    cgmData = data[data['cgmValue'].notnull()]
    cgmData['delta'] = cgmData.apply(lambda row: getTimeDelta(row, startTime), axis=1)
    values = data[data['cgmValue'].notnull()]['cgmValue'].values
    udata.bginitial = values[0]

    bg = calculateBG(df, udata, udata.simlength * 60)

    simbg = bg[0]
    simbg_adv = bg[5]

    ## get last cgm value
    lastValue = cgmData['cgmValue'].values[len(cgmData) -1]
    lastTime = int(cgmData['delta'].values[len(cgmData) - 1])

    ## get second last
    secondlastValue = cgmData['cgmValue'].values[len(cgmData) - 10]
    secondlastTime = int(cgmData['delta'].values[len(cgmData) - 10])

    delta = simbg[secondlastTime] - simbg[lastTime]
    delta_adv = simbg_adv[secondlastTime] - simbg_adv[lastTime]

    prediction = lastValue + delta
    prediction_adv = lastValue + delta_adv

    error = lastValue - prediction
    error_adv = lastValue - prediction_adv


    return [error, error_adv]


def getTimeDelta(row, start):
    if isinstance(start, datetime):
        startTime = start
    else:
        startTime = datetime.strptime(start + timeZone, timeFormat)
    if isinstance(row.datetime, datetime):
        time = row.datetime
    else:
        time = datetime.fromisoformat(row.datetime)


    if time < startTime:
        timeDifference = startTime - time
        return - timeDifference.seconds / 60
    else:
        timeDifference = time - startTime
        return timeDifference.seconds / 60


def checkCurrent(data, udata, startTime):

    events = extractor.getEvents(data)

    converted = events.apply(lambda event: convertTimes(event, startTime))
    df = pandas.DataFrame([vars(e) for e in converted])
    df = df[df.time > 0]
    df = df[df.time < udata.simlength * 60]

    cgmtrue = data[data['cgmValue'].notnull()]
    cgmtrue['delta'] = cgmtrue.apply(lambda row: getTimeDelta(row, startTime), axis=1)
    cgmtrue = cgmtrue[0 < cgmtrue['delta']]
    cgmtrue = cgmtrue[cgmtrue['delta'] < udata.simlength * 60]
    cgmX = cgmtrue['delta'].values
    cgmY = cgmtrue['cgmValue'].values

    initialBG = cgmY[0]

    udata.bginitial = initialBG

    data = calculateBG(df, udata, 500)

    basalValues = df[df.etype == 'tempbasal']
    carbValues = df[df.etype == 'carb']
    bolusValues = df[df.etype == 'bolus']


    for i in range(0,len(data[0])):
        print(str(data[3][i]) + '\t' * 5 + str(data[0][i]) + '\t' * 5 + str(data[1][i]) + '\t' * 5 + str(data[2][i]) )

    dataasdf = pandas.DataFrame([data[1], data[0], data[1], data[2]])

    plt.figure(figsize=(10, 7))
    plt.grid(color="#cfd8dc")
    plt.plot(cgmX, cgmY, "#263238", alpha=0.8, label="real BG")

    plt.plot(data[3], data[0], "#b71c1c", alpha=0.8, label="sim BG")
    plt.plot(data[3], data[5], "#4527a0", alpha=0.8, label="sim BG ADV")

    plt.plot(data[3], data[1], "#64dd17", alpha=0.8, label="sim BGC")
    plt.plot(data[3], data[2], "#d50000", alpha=0.8, label="sim BGI")
    plt.plot(data[3], data[4], "#aa00ff", alpha=0.8, label="sim BGI ADV")

    plt.plot(basalValues.time, [0] * len(basalValues), "bo", alpha=0.8, label="basal event")
    plt.plot(carbValues.time, [0] * len(carbValues),"go", alpha=0.8, label="carb event")
    plt.plot(bolusValues.time, [0] * len(bolusValues),"ro", alpha=0.8, label="bolus (not used)")
    plt.legend(loc=2, bbox_to_anchor=(1,1))
    plt.tight_layout(pad=6)
    plt.savefig("result.png", dpi=300)


def convertTimes(event, start):
    if isinstance(start, datetime):
        startTime = start
    else:
        startTime = datetime.strptime(start + timeZone, timeFormat)

    eventTime = datetime.strptime(event.time + timeZone, timeFormat)
    timeDifference = eventTime - startTime
    if eventTime < startTime:
        timeDifference = startTime - eventTime
        event.time = - timeDifference.seconds / 60
    else:
        event.time = timeDifference.seconds / 60
    if event.etype == "tempbasal":
        t1 = datetime.strptime(event.t1 + timeZone, timeFormat)
        event.t1 = event.time
        #timeDifference = eventTime - t1
        #event.t1 = timeDifference.seconds
        #t2 = datetime.strptime(event.t2, timeFormat)
        #timeDifference = eventTime - t2
        event.t2 = event.t1 + 30 # TODO better estimate
    #print(event.time, event.t1, event.t2)
    return event