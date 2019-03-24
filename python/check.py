import logging
import os
from datetime import datetime

import numpy as np
from matplotlib import gridspec, pyplot as plt

from predictors import optimizer, predict
from PredictionWindow import PredictionWindow
from matplotlib import cm

from predictors.LastNDelta import LastNDelta
from predictors.SameValue import SameValue
from predictors.arima import Arima
from predictors.math_model import MathPredictor
from predictors.optimizer import Optimizer
from predictors.predictor import Predictor

import pandas as pd

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


error_times = np.array([15, 30, 45, 60, 90, 120, 150, 180])





def check_and_plot(pw: PredictionWindow):
    # Set values needed for calculations
    pw.set_values(error_times)
    # If there are no events stop, otherwise there will be errors TODO maby fix
    if pw.events.empty:
        return None

    #predictors = [Optimizer(pw, [30, 60, 90, 120, 240]), Arima(pw), MathPredictor(pw)]
    predictors = [Optimizer(pw, [30, 60, 90, 120, 240]), Arima(pw), SameValue(pw), LastNDelta(pw, 30), LastNDelta(pw, 180)]

    success = list(map(lambda predictor: predictor.calc_predictions(error_times), predictors))

    errors = calculate_errors(predictors, pw)

    if pw.plot:
        graphs = list(map(lambda predictor: predictor.get_graph(), predictors))
        plot_graphs(pw, graphs, errors, predictors)

    logger.info("errors {}".format(errors))
    return errors

    # Run Prediction
    # If plot option is on, we need the whole graph, not only the error checkpoints
    if pw.plot:
        sim_bg, iob, cob = predict.calculateBG(pw.events, pw.userData)

        # Get prediction Value for last train value
        prediction_last_train = np.array([sim_bg[0][pw.time_last_train], sim_bg[5][pw.time_last_train]])
        # logger.debug("prediction train " + str(prediction_last_train))
        # Get prediction Value for last value
        prediction_last_value = np.array([sim_bg[0][pw.time_last_value], sim_bg[5][pw.time_last_value]])
        # logger.debug("prediction value " + str(prediction_last_value))
        # get prediction with optimized parameters
        prediction_optimized, optimized_curve, optimized_carb_events = optimizer.optimize_mix(pw)
        #prediction_optimized_60, optimized_curve_60, optimized_carb_events_60 = optimizer.optimize(pw, 60)
        #prediction_optimized_90, optimized_curve_90, optimized_carb_events_90 = optimizer.optimize(pw, 90)
        #prediction_optimized_120, optimized_curve_120, optimized_carb_events_120 = optimizer.optimize(pw, 120)
        # logger.info("optimizer prediction " + str(prediction_optimized))
    else:
        # Get prediction Value for last train value
        prediction_last_train = np.array(
            predict.calculateBGAt2(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.events, pw.userData))
        # logger.debug("prediction train " + str(prediction_last_train))
        # Get prediction Value for last value
        prediction_last_value = np.array(predict.calculateBGAt2(pw.userData.simlength * 60, pw.events, pw.userData))
        # logger.debug("prediction value " + str(prediction_last_value))
        prediction_optimized = optimizer.optimize_mix(pw)
        #prediction_optimized_60 = optimizer.optimize(pw, 60)
        #prediction_optimized_90 = optimizer.optimize(pw, 90)
        #prediction_optimized_120 = optimizer.optimize(pw, 120)

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
    #prediction = np.append(prediction, prediction_optimized_90)
    #prediction = np.append(prediction, prediction_optimized_120)
    # Get ARIMA prediction
    #prediction_arima, order = arima.get_arima_prediction(pw)
    #pw.prediction = np.append(prediction, prediction_arima.iat[-1])
    # logger.debug("prediction " + str(prediction))
    # calculate error
    pw.prediction = prediction
    pw.errors = np.subtract(pw.lastValue, pw.prediction)
    # logger.debug("errors " + str(errors))

    if pw.plot:
        plot_graph(pw, sim_bg, optimized_curve, optimized_carb_events, None, None, None, None, None, None, prediction30, None, iob, cob)

    return pw.errors.tolist(), None


def setupPlot(ax, pw: PredictionWindow, y_height: int, y_step: int, short: bool = False, negative: bool = False):
    x_start = 0
    if short:
        x_start = pw.userData.train_length()
    x_end = pw.userData.simlength * 60 + 1.
    y_start = 0
    if negative:
        y_start = - y_height - 1
    y_end = y_height + 1
    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_height)
    plt.grid(color = "#cfd8dc")
    major_ticks_x = np.arange(x_start, x_end, 60)
    minor_ticks_x = np.arange(x_start, x_end, 15)
    major_ticks_y = np.arange(y_start, y_end, y_step)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor = True)
    ax.set_yticks(major_ticks_y)
    ax.grid(which = 'minor', alpha = 0.2)
    ax.grid(which = 'major', alpha = 0.5)
    # Plot Line when prediction starts
    plt.axvline(x = pw.userData.train_length(), color = "black")

    plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, left = False)
    plt.box(False)


def calculate_errors(predictors: [], pw: PredictionWindow) -> []:
    errors = []
    for predictor in predictors:
        error = {'predictor': predictor.name,
                 'errors': predictor.prediction_values - pw.real_values}
        errors.append(error)
    return errors

def plotLegend():
    # Plot Legend
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.tight_layout(pad = 6)


def plot_bg_prediction(ax, pw: PredictionWindow, graphs: []):
    setupPlot(ax, pw, 400, 50)
    plt.title("Blood Glucose Level Prediction")
    # Plot real blood glucose readings
    plt.plot(pw.cgmY, alpha = 0.8, label = "real BG")
    for graph in graphs:
        plt.plot(graph['values'], label = graph['label'])

    plotLegend()


def plot_events(ax, pw: PredictionWindow):
    setupPlot(ax, pw, 10, 2)
    plt.title("Events")
    # get events
    basalValues = pw.events[pw.events.etype == 'tempbasal']
    carbValues = pw.events[pw.events.etype == 'carb']
    bolusValues = pw.events[pw.events.etype == 'bolus']
    # Plot Events
    # logger.debug(basalValues.values[0])
    if not basalValues.empty:
        plt.bar(basalValues.time, basalValues.dbdt, 5, alpha = 0.8, label = "basal event")
    # logger.debug(carbValues)
    if not carbValues.empty:
        plt.bar(carbValues.time, carbValues.grams, 5, alpha = 0.8, label = "carb event")
    if not bolusValues.empty:
        plt.bar(bolusValues.time, bolusValues.units, 5, alpha = 0.8, label = "bolus event")
    plotLegend()


def plot_events_optimized(ax, pw, predictors):
    opt = None
    for predictor in predictors:
        if isinstance(predictor, Optimizer):
            opt = predictor
            break
    if not opt:
        return
    setupPlot(ax, pw, 10, 2)
    plt.title("Events Optimized")
    basalValues = opt.all_events[opt.all_events.etype == 'tempbasal']
    carbValues = opt.all_events[opt.all_events.etype == 'carb']
    bolusValues = opt.all_events[opt.all_events.etype == 'bolus']
    # Plot Events
    # logger.debug(basalValues.values[0])
    if not basalValues.empty:
        plt.bar(basalValues.time, basalValues.dbdt, 5, alpha = 0.8, label = "basal event")
    # logger.debug(carbValues)
    if not carbValues.empty:
        plt.bar(carbValues.time, carbValues.grams, 5, alpha = 0.8, label = "carb event")
    if not bolusValues.empty:
        plt.bar(bolusValues.time, bolusValues.units, 5, alpha = 0.8, label = "bolus event")
    plotLegend()


def plot_iob_cob(ax, pw, predictors: [Predictor]):
    opt = None
    for predictor in predictors:
        if isinstance(predictor, Optimizer):
            opt = predictor
            break
    if not opt:
        return
    plt.title("IOB, COB Optimized")
    setupPlot(ax, pw, 2, 0.4)
    plt.plot(opt.iob, label="Insulin on Board")
    plt.plot(opt.cob, label="Carbs on Board")
    plotLegend()


def plot_graph_prediction(ax, pw, graphs):
    setupPlot(ax, pw, 400, 50, short=True)
    plt.title("Blood Glucose Level Prediction")
    # Plot real blood glucose readings
    plt.plot(pw.cgmY[pw.userData.train_length():], alpha = 0.8, label = "real BG")
    for graph in graphs:
        if 'optimizer' in graph['label']:
            plt.plot(pd.Series(graph['values'])[pw.userData.train_length():], label = graph['label'])
        else:
            plt.plot(graph['values'], label = graph['label'])
    plotLegend()


def plot_errors(ax, pw, errors):
    setupPlot(ax, pw, 400, 50, short = True, negative = True)
    plt.title("Errors")
    positions = iter([-6, -3, 0, 3, 6])
    for error in errors:
        plt.bar(error['errors'].index + next(positions), error['errors'].tolist(), 3, alpha = 0.5,
                label = error['predictor'])
    plotLegend()



def plot_graphs(pw: PredictionWindow, graphs, errors, predictors: [Predictor]):
    # set figure size
    fig = plt.figure(figsize = (20, 16))
    gs = gridspec.GridSpec(6, 1, height_ratios = [3, 1, 1, 1, 3, 3])
    subplot_iterator = iter(gs)

    # BLOOD GLUCOSE PREDICTION
    plot_bg_prediction(plt.subplot(next(subplot_iterator)), pw, graphs)

    # EVENTS ORIGINAL
    plot_events(plt.subplot(next(subplot_iterator)), pw)

    # EVENTS OPTIMIZED
    plot_events_optimized(plt.subplot(next(subplot_iterator)), pw, predictors)

    # IOB / COB
    plot_iob_cob(plt.subplot(next(subplot_iterator)), pw, predictors)

    # PREDICTION PLOT
    plot_graph_prediction(plt.subplot(next(subplot_iterator)), pw, graphs)

    # ERRORS
    plot_errors(plt.subplot(next(subplot_iterator)), pw, errors)

    # SAVE PLOT TO FILE
    plt.savefig(path + "results/plots/result-n-" + pw.startTime.strftime('%Y-%m-%d-%H-%M') + ".png", dpi = 150)
    plt.close()

def plot_graph(pw: PredictionWindow, sim_bg, optimized_curve, optimized_carb_events, optimized_curve_60, optimized_carb_events_60, optimized_curve_90, optimized_carb_events_90, optimized_curve_120, optimized_carb_events_120, prediction30, arima_values, iob,
               cob):
    # get values for prediction timeframe
    prediction_vals = getPredictionVals(pw, sim_bg[0])
    prediction_vals_adv = getPredictionVals(pw, sim_bg[5])

    # get events
    basalValues = pw.events[pw.events.etype == 'tempbasal']
    carbValues = pw.events[pw.events.etype == 'carb']
    bolusValues = pw.events[pw.events.etype == 'bolus']

    # set figure size
    fig = plt.figure(figsize = (10, 16))
    gs = gridspec.GridSpec(5, 1, height_ratios = [3, 3, 3, 1, 1])
    # fig, ax = plt.subplots()
    subplot_iterator = iter(gs)

    # ------------------------- GRAPHS Model --------------------------------------------
    ax = plt.subplot(next(subplot_iterator))
    setupPlot(ax, pw, 400, 50)

    # Plot real blood glucose readings
    plt.plot(pw.cgmY, alpha = 0.8, label = "real BG")
    # Plot sim results
    plt.plot(sim_bg[3], sim_bg[0], alpha = 0.5, label = "sim BG")
    plt.plot(range(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.userData.simlength * 60),
             prediction_vals, alpha = 0.8, label = "SIM BG Pred")
    plt.plot(sim_bg[3], sim_bg[5], alpha = 0.5, label = "sim BG ADV")
    plt.plot(range(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.userData.simlength * 60),
             prediction_vals_adv, alpha = 0.8, label = "SIM BG Pred ADV")

    # Plot Legend
    plotLegend()
    # ------------------------- GRAPHS Arima SV L30 --------------------------------------------
    ax = plt.subplot(next(subplot_iterator))
    setupPlot(ax, pw, 400, 50)

    # Plot real blood glucose readings
    plt.plot(pw.cgmY, alpha = 0.8, label = "real BG")
    # Same value prediction
    plt.axhline(y = prediction_vals[0],
                xmin = (pw.userData.train_length() / 60) / pw.userData.simlength, alpha = 0.8,
                label = "Same Value Prediction")
    # last 30 prediction value
    plt.plot([pw.userData.train_length(), pw.userData.simlength * 60], [pw.train_value, prediction30],
             alpha = 0.8, label = "Last 30 Prediction")

    # arima prediction
    #index = np.arange(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.userData.simlength * 60 + 1,
    #                  pw.userData.predictionlength / (len(arima_values) - 1))
    #arima_values.index = index
    #plt.plot(arima_values, alpha = 0.8, label = "arima prediction")

    # Plot Legend
    plotLegend()
    # ------------------------- GRAPHS Optimized --------------------------------------------
    ax = plt.subplot(next(subplot_iterator))
    setupPlot(ax, pw, 400, 50)

    # Plot real blood glucose readings
    plt.plot(pw.cgmY, alpha = 0.8, label = "real BG")

    # optimized prediction
    #plt.plot(optimized_curve_60, alpha = 0.8, label = "optimized curve 60")
    #plt.plot(optimized_curve_90, alpha = 0.8, label = "optimized curve 90")
    #plt.plot(optimized_curve_120, alpha = 0.8, label = "optimized curve 120")
    plt.plot(optimized_curve, alpha = 0.8, label = "optimized curve mix")

    # Plot Legend
    plotLegend()

    # ---------------------------- EVENTS -----------------------------------------
    ax = plt.subplot(next(subplot_iterator))
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
    #if not optimized_carb_events_60.empty:
    #    plt.bar(optimized_carb_events_60.index, optimized_carb_events_60, 5, alpha = 0.8, label = "optimized carb event")
    if not optimized_carb_events.empty:

        ctypes = optimized_carb_events.ctype.unique()
        colors = iter(cm.hsv(np.linspace(0,1,ctypes.size+1)))
        positions = iter([-5.5, -3,0, 3, 5.5])
        for ctype in ctypes:
            events = optimized_carb_events[optimized_carb_events.ctype == ctype]
            color = next(colors)
            position = next(positions)
            plt.bar(events.time + position, events.grams, 2, alpha = 0.8,
                    label = "optimized carb event mixed: {}".format(ctype), color=color)

    plotLegend()
    plt.subplots_adjust(hspace = 0.2)
    # ---------------------------- IOB COB -----------------------------------------
    ax = plt.subplot(next(subplot_iterator))
    setupPlot(ax, pw, 3, 0.5)

    plt.plot(iob, label = "Insulin on Board")
    plt.plot(cob, label = "Carbs on Board")

    plotLegend()
    plt.subplots_adjust(hspace = 0.2)
    # ---------------------------------------------------------------------

    # Save plot as svgz (smallest format, able to open with chrome)
    plt.savefig(path + "results/plots/result-" + pw.startTime.strftime('%Y-%m-%d-%H-%M') + ".png", dpi = 150)
    plt.close()
