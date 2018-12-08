import pandas

from autotune_prep import writeJson
from check import checkCurrent
from Classes import Event, UserData
from readData import read_data
from predict import calculateBG
from rolling import predictRolling, predictRollingCached
import json
import logging

logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)

filename = "../../Data/data0318.csv"
filename17 = "../../Data/data2017.csv"
filename1217 = "../../Data/data1217.csv"
filenameDocker = "/t1d/data/data1217.csv"

def mainplt():
    uevent = [Event.createBolus(time=10, units=3.8),
              Event.createCarb(time=5, grams=18, ctype=180),
              Event.createTemp(time=0,dbdt=2,t1=10,t2=20)]

    udata = UserData(bginitial= 0, cratio= 0.2, idur= 3, inputeeffect= 5, sensf = 10, simlength = 1, stats = None)

    calculateBG(uevent, udata, 500)

def mainPredict():
    data = read_data(filename)
    data = data[data['date'] == "08.03.18"]
    udata = UserData(bginitial=0, cratio=5, idur=3, inputeeffect=None, sensf=41, simlength=12, stats=None)
    res = checkCurrent(data, udata)


def mainFast():
    # get rolling prediction window
    logger.info("Start Main")
    logger.debug("Load Data")
    data = read_data(filenameDocker)
    logger.debug("Loaded Data with shape: " + str(data.shape))
    udata = UserData(bginitial=0, cratio=0.08, idur=4, inputeeffect=None, sensf=5, simlength=4, stats=None)
    #res = predictRolling(data, udata)
    res = predictRollingCached(data, udata)

    res_series = pandas.Series(res)
    mean = res_series.mean(skipna=True)
    median = res_series.median(skipna=True)
    numberofna = res_series.isna().sum()
    jsonobject = {"mean": int(mean), "median": int(median), "number of NA": int(numberofna), "data": res}
    file = open('./result.json', 'w')
    file.write(json.dumps(jsonobject))
    logger.info("finished")

def main():
    # get rolling prediction window
    logger.info("Start Main")
    logger.debug("Load Data")
    data = read_data(filename17)
    logger.debug("Loaded Data with shape: " + str(data.shape))
    udata = UserData(bginitial=0, cratio=0.08, idur=4, inputeeffect=None, sensf=5, simlength=4, stats=None)
    res = predictRolling(data, udata)


    logger.info("finished")

if __name__ == '__main__':
    mainPredict()