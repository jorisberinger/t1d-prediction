import pandas

from analyze import analyze
from autotune_prep import writeJson
from check import checkCurrent
from Classes import Event, UserData
from readData import read_data
import rolling
import predict
import autotune_prep
import autotune
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

# makes a 1 hour prediction based on a 5 hour data set. it will generate a plot to show prediction in result.png
def mainPredict(data, userdata):
    # select date and start time for prediction
    date = "21.12.17"
    time = "18:00"
    autotune_res = autotune.getSensAndCR(date)
    userdata.cratio = autotune_res["cr"]
    logger.info(autotune_res)
    userdata.sensf = autotune_res["sens"][0]["sensitivity"]
    data = data[data['date'] == date]
    start = date+","+time
    res = checkCurrent(data, userdata, start)


def predictFast(data, userdata):
    steps = 5
    res = rolling.predictRollingCached(data, userdata, steps)
    res_series = pandas.Series(res[0])
    res_adv_series = pandas.Series(res[1])
    mean = res_series.mean(skipna=True)
    median = res_series.median(skipna=True)
    mean_adv = res_adv_series.mean(skipna=True)
    median_adv = res_adv_series.median(skipna=True)
    jsonobject = {"mean": float(mean), "mean_adv": float(mean_adv), "median": float(median),  "median_adv": int(median_adv), "data": res}
    filename = "result-" + str(steps) + ".json"
    analyze(jsonobject, filename)
    file = open(filename, 'w')
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
    data = read_data(filenameDocker)
    #data = read_data(filename1217)
    logger.debug("Loaded Data with shape: " + str(data.shape))
    logger.debug("set user data")
    udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=6, stats=None)
    logger.debug("Prep glucose and insulin history for autotune as json")
    autotune_prep.prep_for_autotune(data)
    logger.debug("Run autotune")
    autotune.run_autotune(data)
    logger.debug("Run Prediciton")
    mainPredict(data, udata)
   # predictFast(data, udata)
    logger.info("finished!")

if __name__ == '__main__':
    main()


# prediction mit gleichem wert
# prediction mit steigung von  letzten 3 werten
# verticale linie in result.png und prediction plotten
# prediction time 60 min
# 6h
# basal raten -> bolus
# autotune fuer jeden tag
# verschiedene zeitfenster fuer autotune
# carb platzierung
# carbs alle 10 minuten, unit optimieren bis die kurve passt
