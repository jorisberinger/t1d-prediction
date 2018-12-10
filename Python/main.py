import pandas

from autotune_prep import writeJson
from check import checkCurrent
from Classes import Event, UserData
from readData import read_data
import rolling
import predict
import json
import logging

logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)

filename = "../../Data/data0318.csv"
filename17 = "../../Data/data2017.csv"
filename1217 = "../../Data/data1217.csv"
filenameDocker = "/t1d/data/data1217.csv"

def plt(udata):
    uevent = [Event.createBolus(time=10, units=3.8),
              Event.createCarb(time=5, grams=18, ctype=180),
              Event.createTemp(time=0,dbdt=2,t1=10,t2=20)]

    predict.calculateBG(uevent, udata, 500)

def mainPredict(data, userdata):
    data = data[data['date'] == "15.12.17"]
    start = "15.12.17,11:00"
    res = checkCurrent(data, userdata, start)


def predictFast(data, userdata):
    res = rolling.predictRollingCached(data, userdata)

    res_series = pandas.Series(res[0])
    res_adv_series = pandas.Series(res[1])
    mean = res_series.mean(skipna=True)
    median = res_series.median(skipna=True)
    mean_adv = res_adv_series.mean(skipna=True)
    median_adv = res_adv_series.median(skipna=True)
    jsonobject = {"mean": float(mean), "mean_adv": float(mean_adv),"median": float(median),  "median_adv": int(median_adv), "data": res}
    file = open('./result-10.json', 'w')
    file.write(json.dumps(jsonobject))
    logger.info("finished")

def predictRolling(data, userdata):
    rolling.predictRolling(data, userdata)

def compareIOBs(userdata):
    # create sample events
    uevents = [Event.createBolus(30, 1), Event.createBolus(180, 1)]
    predict.compareIobs(userdata, uevents, "compare.png")

def main():
    # get rolling prediction window
    logger.info("Start Main")
    logger.debug("Load Data")
    #data = read_data(filenameDocker)
    data = read_data(filename1217)
    logger.debug("Loaded Data with shape: " + str(data.shape))
    udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=5, stats=None)
    logger.debug("set user data")

    predictFast(data, udata)
    logger.info("finished")

if __name__ == '__main__':
    main()