import extractor
from datetime import datetime

from Classes import UserData
from predict import calculateBG
from matplotlib import pyplot as plt

timeFormat = "%d.%m.%y,%H:%M"

def checkCurrent(data):
    events = extractor.getEvents(data)

    converted = events.apply(lambda event: convertTimes(event, "08.03.18,09:00"))
    dataZero = data[data['time'] == "08:58"]
    initialBG = dataZero['cgmValue'].values[0]


    udata = UserData(bginitial=initialBG, cratio=10, idur=5, inputeeffect=1, sensf=1, simlength=20, stats=None)
    data = calculateBG(converted.values, udata, 100)
    print(data)
    plt.plot(data[0])
    plt.plot(data[1])
    plt.plot(data[2])
    plt.show()

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
        event.t1 = timeDifference.seconds
        #t2 = datetime.strptime(event.t2, timeFormat)
        #timeDifference = eventTime - t2
        event.t2 = event.t1 + 2 * 60
    print(event.time, event.t1, event.t2)
    return event