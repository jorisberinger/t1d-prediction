import logging
import os
import time
import analyze
import autotune
import autotune_prep
import gifmaker
import rolling
from Classes import UserData
from readData import read_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data.csv"



# make prediction every 15 minutes
def predictFast(data, userdata, autotune_res, plotOption):
    # make a rolling prediction
    res = rolling.predictRolling(data, userdata, autotune_res, plotOption)
    # analyse data and prepare for output
    summary = analyze.getSummary(res)
    analyze.createErrorPlots(summary)

    gifmaker.makeGif("/t1d/results/plots/", data)
    logger.info("finished prediction")


def runAutotune(data):
    logger.debug("Prep glucose and insulin history for autotune as json")
    autotune_prep.prep_for_autotune(data)
    logger.debug("Run autotune")
    res = autotune.run_autotune(data)
    return res

def main():
    # SELECT OPTIONS
    run_autotune = False   # Select True if autotune should run. If data set has been run before, set to False to improve speed.
    create_plots = False  # Select True if you want a plot for every prediction window

    logger.info("Start Main!")

    data = read_data(filename)
    udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=11, predictionlength=60, stats=None)

    logger.info("Run Autotune? " + str(run_autotune))
    if run_autotune:
        autotune_res = runAutotune(data)
    else:
        autotune_res = autotune.getAllSensAndCR(data)

    logger.debug("Run Prediciton")
    predictFast(data, udata, autotune_res, create_plots)

    logger.info("Main finished!")


if __name__ == '__main__':
    start_time = time.process_time()
    main()
    logger.info(str(time.process_time() - start_time) + " seconds")
