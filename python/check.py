import logging
import os
from datetime import datetime

import numpy as np
from matplotlib import gridspec, pyplot as plt

import arima
import extractor
import optimizer
import predict
from Classes import PredictionWindow

logger = logging.getLogger(__name__)

timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"

path = os.getenv('T1DPATH', '../')


def getTimeDelta(row, start):
    time = row.datetime
    if time < start:
        timeDifference = start - time
        return - timeDifference.seconds / 60
    else:
        timeDifference = time - start
        return timeDifference.seconds / 60


def getClosestIndex(cgmY, number):
    smaller = cgmY[cgmY.index <= number]
    closest = smaller.index[-1]
    return closest


def getPredictionVals(pw: PredictionWindow, simData):
    zero = simData[pw.userData.simlength * 60 - pw.userData.predictionlength]
    start = pw.cgmY[pw.userData.simlength * 60 - pw.userData.predictionlength]
    res = []
    for i in range(int(pw.userData.simlength * 60 - pw.userData.predictionlength), int(pw.userData.simlength * 60)):
        res.append(start + simData[i] - zero)
    return res


def getPredictionAt(index_last_train, df_train, udata):
    prediction = predict.calculateBGAt(index_last_train, df_train, udata, udata.simlength * 60)
    return prediction


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

    if event.etype == "carb":
        event.time = event.time - 30  # TODO verify
    if event.etype == "tempbasal":
        event.t1 = event.time
        event.t2 = eventTime
    return event


def getCgmReading(data):
    cgmtrue = data[data['cgmValue'].notnull()]
    if len(cgmtrue) > 0:
        cgmY = cgmtrue['cgmValue']
        return cgmY
    else:
        return None


def checkAndPlot(pw: PredictionWindow):
    # Get all Values of the Continuous Blood Glucose Reading, cgmX as TimeDelta from Start and cgmY the paired Value
    pw.cgmY = getCgmReading(pw.data)

    # Set initial Blood Glucose Level
    pw.userData.bginitial = pw.cgmY[0]

    # Get Train Datapoint
    pw.train_value = pw.data.at[float(pw.userData.simlength * 60 - pw.userData.predictionlength), 'cgmValue']
    # Get last Datapoint
    pw.lastValue = pw.data.at[float(pw.userData.simlength * 60), 'cgmValue']

    # Get Events for Prediction
    pw.events = extractor.getEventsAsDataFrame(pw)
    if pw.events.empty:
        return None

    # Run Prediction
    if pw.plot:
        sim_bg, iob, cob = predict.calculateBG(pw.events, pw.userData)
        # logger.info("IOB")
        # logger.info(iob)
        # logger.info("COB")
        # logger.info(cob)
        plt.plot(iob, label = "iob")
        plt.plot(cob, label = "cob")
        plt.legend()
        plt.show()
        # Get prediction Value for last train value
        prediction_last_train = np.array([sim_bg[0][pw.time_last_train], sim_bg[5][pw.time_last_train]])
        # logger.debug("prediction train " + str(prediction_last_train))
        # Get prediction Value for last value
        prediction_last_value = np.array([sim_bg[0][pw.time_last_value], sim_bg[5][pw.time_last_value]])
        # logger.debug("prediction value " + str(prediction_last_value))
        # get prediction with optimized parameters
        prediction_optimized, optimized_curve, optimized_carb_events = optimizer.optimize(pw)
        # logger.info("optimizer prediction " + str(prediction_optimized))
    else:
        # Get prediction Value for last train value
        prediction_last_train = np.array(
            predict.calculateBGAt2(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.events, pw.userData))
        # logger.debug("prediction train " + str(prediction_last_train))
        # Get prediction Value for last value
        prediction_last_value = np.array(predict.calculateBGAt2(pw.userData.simlength * 60, pw.events, pw.userData))
        # logger.debug("prediction value " + str(prediction_last_value))
        prediction_optimized = optimizer.optimize(pw)
        # logger.info("optimizer prediction " + str(prediction_optimized))

    # Get Delta between train and last value
    prediction_delta = prediction_last_value - prediction_last_train
    # logger.debug("delta " + str(prediction_delta))
    # add on last Train value
    prediction = np.add(pw.train_value, prediction_delta)
    # logger.debug("prediction " + str(prediction))
    # add same value prediction
    prediction = np.append(prediction, pw.train_value)

    # logger.debug("prediction " + str(prediction))
    # add last 30 min prediction
    last30delta = pw.train_value - pw.cgmY[pw.userData.simlength * 60 - pw.userData.predictionlength - 30]
    prediction30delta = last30delta * pw.userData.predictionlength / 30
    prediction30 = pw.train_value + prediction30delta
    prediction = np.append(prediction, prediction30)
    prediction = np.append(prediction, prediction_optimized)
    # Ger ARIMA prediction
    prediction_arima, order = arima.get_arima_prediction(pw)
    pw.prediction = np.append(prediction, prediction_arima.iat[-1])
    # logger.debug("prediction " + str(prediction))
    # calculate error
    pw.errors = np.subtract(pw.lastValue, pw.prediction)
    # logger.debug("errors " + str(errors))

    if pw.plot:
        plot_graph(pw, sim_bg, optimized_curve, optimized_carb_events, prediction30, prediction_arima)

    return pw.errors.tolist(), order


def setupPlot(ax, pw: PredictionWindow, y_height: int, y_step: int):
    plt.xlim(0, pw.userData.simlength * 60 + 1)
    plt.ylim(0, y_height)
    plt.grid(color = "#cfd8dc")
    major_ticks_x = np.arange(0, pw.userData.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, pw.userData.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, y_height + 1, y_step)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor = True)
    ax.set_yticks(major_ticks_y)
    ax.grid(which = 'minor', alpha = 0.2)
    ax.grid(which = 'major', alpha = 0.5)
    # Plot Line when prediction starts
    plt.axvline(x = (pw.userData.simlength - 1) * 60, color = "black")

    plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False)
    plt.box(False)


def plotLegend():
    # Plot Legend
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.tight_layout(pad = 6)


def plot_graph(pw: PredictionWindow, sim_bg, optimized_curve, optimized_carb_events, prediction30, arima_values):
    # get values for prediction timeframe
    prediction_vals = getPredictionVals(pw, sim_bg[0])
    prediction_vals_adv = getPredictionVals(pw, sim_bg[5])

    # get events
    basalValues = pw.events[pw.events.etype == 'tempbasal']
    carbValues = pw.events[pw.events.etype == 'carb']
    bolusValues = pw.events[pw.events.etype == 'bolus']

    # set figure size
    fig = plt.figure(figsize = (10, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1])
    # fig, ax = plt.subplots()

    # ------------------------- GRAPHS --------------------------------------------
    ax = plt.subplot(gs[0])
    setupPlot(ax, pw, 400, 50)

    # Plot real blood glucose readings
    plt.plot(pw.cgmY, "#263238", alpha = 0.8, label = "real BG")
    # Plot sim results
    plt.plot(sim_bg[3], sim_bg[0], "#b71c1c", alpha = 0.5, label = "sim BG")
    plt.plot(range(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.userData.simlength * 60),
             prediction_vals, "#b71c1c", alpha = 0.8, label = "SIM BG Pred")
    plt.plot(sim_bg[3], sim_bg[5], "#4527a0", alpha = 0.5, label = "sim BG ADV")
    plt.plot(range(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.userData.simlength * 60),
             prediction_vals_adv, "#4527a0", alpha = 0.8, label = "SIM BG Pred ADV")
    # Same value prediction
    plt.axhline(y = prediction_vals[0],
                xmin = (pw.userData.simlength - pw.userData.predictionlength / 60) / pw.userData.simlength, alpha = 0.8,
                label = "Same Value Prediction")
    # last 30 prediction value
    plt.plot([(pw.userData.simlength - 1) * 60, pw.userData.simlength * 60], [pw.train_value, prediction30], "#388E3C",
             alpha = 0.8, label = "Last 30 Prediction")
    # optimized prediction
    plt.plot(optimized_curve, alpha = 0.8, label = "optimized curve")
    # arima prediction
    index = np.arange(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.userData.simlength * 60 + 1,
                      pw.userData.predictionlength / (len(arima_values) - 1))
    arima_values.index = index
    plt.plot(arima_values, alpha = 0.8, label = "arima prediction")
    # Plot Legend
    plotLegend()

    # ---------------------------- EVENTS -----------------------------------------
    ax = plt.subplot(gs[1])
    setupPlot(ax, pw, 10, 2)

    # Plot Events
    # logger.debug(basalValues.values[0])
    if not basalValues.empty:
        plt.bar(basalValues.time, basalValues.dbdt, 5, alpha = 0.8, label = "basal event (not used)")
    # logger.debug(carbValues)
    if not carbValues.empty:
        plt.bar(carbValues.time, carbValues.grams, 5, alpha = 0.8, label = "carb event")
    if not bolusValues.empty:
        plt.bar(bolusValues.time, bolusValues.units, 5, alpha = 0.8, label = "bolus event")
    if not optimized_carb_events.empty:
        plt.bar(optimized_carb_events.index, optimized_carb_events, 5, alpha = 0.8, label = "optimized carb event")

    plotLegend()
    plt.subplots_adjust(hspace = 0.2)
    # ---------------------------------------------------------------------

    # Save plot as svgz (smallest format, able to open with chrome)
    plt.savefig(path + "results/plots/result-" + pw.startTime.strftime('%Y-%m-%d-%H-%M') + ".png", dpi = 150)
    plt.close()
