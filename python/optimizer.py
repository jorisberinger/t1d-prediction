import logging
from datetime import timedelta

import pandas
from scipy.optimize import minimize, Bounds
import check
import extractor
import readData
import rolling
import predict
from Classes import UserData, Event
from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

filenameDocker = "/t1d/data/csv/data-o.csv"

udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=5, predictionlength=60,
                     stats=None)

events = None
def optimize():
    logger.info("start optimizing")

    # load data and select time frame
    data = readData.read_data(filenameDocker)
    data["datetimeIndex"] = data.apply(lambda row: rolling.convertTime(row.date + ',' + row.time), axis=1)
    data["datetime"] = data["datetimeIndex"]
    data = data.set_index('datetimeIndex')
    startTime = data.index[0]
    subset = data.loc[startTime <= data.index]
    subset = subset.loc[startTime + timedelta(hours=5) > subset.index]
    global cgmX
    global cgmY
    cgmX , cgmY = check.getCgmReading(subset, startTime)

    # Extract events
    events = extractor.getEvents(subset)
    converted = events.apply(lambda event: check.convertTimes(event, startTime))
    global df
    df = pandas.DataFrame([vars(e) for e in converted])

    # Remove Carb events and tempbasal
    df = df[df['etype'] != "carb"]
    df = df[df['etype'] != "tempbasal"]

    events = df

    x0 = np.array([1/12] * 20)
    bounds = ((0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20),(0, 20))
    logger.info("bounds " + str(bounds))

    logger.debug(x0)
    predicter(x0)
    #res = minimize(predicter, x0, method='nelder-mead', options = {'xtol': 20, 'maxiter': 200, 'disp': True})
    #res = minimize(predicter, x0, method='L-BFGS-B', bounds=bounds,options = {'ftol': 20, 'maxiter': 500, 'disp': True})
    res = minimize(predicter, x0, bounds=bounds, options = {'disp': True})
    logger.debug(res.x)

    logger.debug(df)

def predicter(inputs):
    logger.info(inputs)
    carbEvents = []
    for i in range(0,len(inputs)):
        carbEvents.append(Event.createCarb(i*15, inputs[i], 60))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])
    #logger.info(ev)
    allEvents = pandas.concat([df,ev])
    #logger.info(allEvents)

    sim = predict.calculateBG(allEvents, udata)
    error = 0

    for i in range(0, 280, 15):
        #logger.info(i)
        #logger.info(i/5)
        #logger.info(cgmX[int(i/5)])

        error += abs(sim[5][i] - cgmY[int(i/5)])

    #logger.info(error)
    logger.info(error)
    return error



if __name__ == '__main__':
    optimize()