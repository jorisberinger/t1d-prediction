import cProfile
import logging
import os
import time
import analyze
from autotune import autotune
from autotune import autotune_prep
import rolling
from Classes import UserData
import readData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data-o3.csv"



# make prediction every 15 minutes
def predictFast(data, userdata, autotune_res, plotOption):
    # make a rolling prediction
    res = rolling.predictRolling(data, userdata, autotune_res, plotOption)
    # analyse data and prepare for output
    summary = analyze.getSummary(res)
    analyze.createErrorPlots(summary)

    #gifmaker.makeGif("/t1d/results/plots/", data)
    logger.info("finished prediction")




def main():
    # SELECT OPTIONS
    run_autotune = False   # Select True if autotune should run. If data set has been run before, set to False to improve speed.
    create_plots = False  # Select True if you want a plot for every prediction window

    logger.info("Start Main!")

    data = readData.read_data(filename)
    udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=11, predictionlength=60, stats=None)

    logger.info("Run Autotune? " + str(run_autotune))
    if run_autotune:
        autotune_res = autotune.runAutotune(data)
    else:
        autotune_res = autotune.getAllSensAndCR(data)

    logger.debug("Run Prediciton")
    predictFast(data, udata, autotune_res, create_plots)

    logger.info("Main finished!")


if __name__ == '__main__':
    start_time = time.process_time()
    profile = False
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    main()
    if profile:
        pr.disable()
        pr.dump_stats(path + "results/profiler/profile")
    logger.info(str(time.process_time() - start_time) + " seconds")
