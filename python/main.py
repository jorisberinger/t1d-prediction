import analyze
import check
from Classes import UserData
from readData import read_data
import rolling
import autotune_prep
import autotune
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

filenameDocker = "/t1d/data/csv/data.csv"



# makes a 1 hour prediction based on a 5 hour data set. it will generate a plot to show prediction in result.png
def createOnePlot(data, userdata):
    # select date and start time for prediction
    date = "21.12.17"
    time = "18:00"
    autotune_res = autotune.getSensAndCR(date)
    userdata.cratio = autotune_res["cr"]
    logger.info(autotune_res)
    userdata.sensf = autotune_res["sens"][0]["sensitivity"]
    data = data[data['date'] == date]
    start = date+","+time
    res = check.checkCurrent(data, userdata, start)


# make prediction every 15 minutes
def predictFast(data, userdata):
    # make a rolling prediction
    res = rolling.predictRolling(data, userdata)
    # analyse data and prepare for output
    summary = analyze.getSummary(res)
    analyze.createErrorPlots(summary)
    logger.info("finished prediction")


def runAutotune(data):
    logger.debug("Prep glucose and insulin history for autotune as json")
    autotune_prep.prep_for_autotune(data)
    logger.debug("Run autotune")
    autotune.run_autotune(data)


def main():
    run_autotune = True   # Select True if autotune should run. If data set has been run before, set to False to improve speed.
    logger.info("Start Main!")
    logger.debug("Load Data")
    data = read_data(filenameDocker)
    udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=6, predictionlength=60, stats=None)

    logger.info("Run Autotune? " + run_autotune)
    if run_autotune:
        runAutotune(data)

    logger.debug("Run Prediciton")
    predictFast(data, udata)
    logger.info("finished!")


if __name__ == '__main__':
    start_time = time.process_time()
    main()
    logger.info(str(time.process_time() - start_time) + " seconds")

