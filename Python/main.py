import pandas
from analyze import analyze
from check import checkCurrent
from Classes import Event, UserData
from readData import read_data
import rolling
import predict
import autotune_prep
import autotune
import json
import logging
import profile
import pstats
from pstats import SortKey

logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)

filenameDocker = "/t1d/data/csv/data.csv"
run_autotune = False


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

# make prediction every 15 minutes
def predictFast(data, userdata):
    setNumber = 1  # for debug
    res = rolling.predictRolling(data, userdata)

    logger.debug(res)
    # analyse data and prepare for output
    df = pandas.DataFrame(res)
    df = df.apply(abs)
    logger.debug(df)
    res_series = pandas.Series(df[0])
    logger.debug(res_series)
    res_series = res_series.apply(abs)
    logger.debug(res_series)
    res_adv_series = pandas.Series(df[1])
    logger.debug(res_adv_series)
    res_same_value = pandas.Series(df[2])
    logger.debug(res_same_value)
    mean = res_series.mean(skipna=True)
    logger.debug(mean)
    median = res_series.median(skipna=True)
    logger.debug(median)
    mean_adv = res_adv_series.mean(skipna=True)
    logger.debug(mean_adv)
    median_adv = res_adv_series.median(skipna=True)
    logger.debug(median_adv)
    mean_same_value = res_same_value.mean(skipna=True)
    logger.debug(mean_same_value)
    median_same_value = res_same_value.median(skipna=True)
    logger.debug(median_same_value)
    logger.info("Results: mean: " + str(mean)+ "\tmean_adv: " + str(mean_adv) + "\tmean_same_value: " + str(mean_same_value)
                + "\tmedian: " + str(median) + "\tmedian_adv: " + str(median_adv) + "\tmedian_same_value: " + str(median_same_value))
    jsonobject = {"mean": float(mean), "mean_adv": float(mean_adv), "mean_same_value": float(mean_same_value),
                  "median": float(median),  "median_adv": float(median_adv), "median_same_value": float(median_same_value),
                  "data": res}
    filename = "/t1d/result-" + str(setNumber) + ".json"
    analyze(jsonobject, filename)
    file = open(filename, 'w')
    file.write(json.dumps(jsonobject))
    logger.info("finished prediction")

def predictRolling(data, userdata):
    rolling.predictRolling(data, userdata)

def compareIOBs(userdata):
    # create sample events
    uevents = [Event.createBolus(30, 1), Event.createBolus(180, 1)]
    predict.compareIobs(userdata, uevents, "/t1d/results/compare.png")

def main():
    # get rolling prediction window
    logger.info("Start Main!")
    logger.debug("Load Data")
    data = read_data(filenameDocker)
    #data = read_data(filename1217)
    logger.debug("Loaded Data with shape: " + str(data.shape))
    logger.debug("set user data")
    udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=6, predictionlength=60, stats=None)
    if run_autotune:
        logger.debug("Prep glucose and insulin history for autotune as json")
        autotune_prep.prep_for_autotune(data)
        logger.debug("Run autotune")
        autotune.run_autotune(data)
    logger.debug("Run Prediciton")
   # mainPredict(data, udata)
    predictFast(data, udata)
    logger.info("finished!")

if __name__ == '__main__':
    prof = profile.run('main()', "/t1d/results/profile")
    p = pstats.Stats("/t1d/results/profile")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).reverse_order().dump_stats("/t1d/results/profile.txt")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(15)
    p.strip_dirs().sort_stats(SortKey.CALLS).print_stats(15)
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(15)


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
