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
from predictors.lstm import LSTM_predictor
from predictors.mean_predictor import Mean_predictor
from predictors.error_predictor import Error_predictor
import pandas as pd

logger = logging.getLogger(__name__)

timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"

path = os.getenv('T1DPATH', '../')


error_times = np.array([15, 30, 45, 60, 90, 120, 150, 180])
colors = ['#005AA9','#0083CC','#009D81','#99C000','#C9D400','#FDCA00','#F5A300','#EC6500','#E6001A','#A60084','#721085']


def check_and_plot(pw: PredictionWindow, item):
    # Set values needed for calculations
    pw.set_values(error_times)
    # If there are no events stop, otherwise there will be errors, same condition if events in prediction time frame
    if pw.events.empty:
        return None

    # Select with predictors should run
    predictors =   [
                    # Optimizer(pw, [15, 30, 60, 90, 120, 240]),
                    Optimizer(pw, [90, 120, 240]),
                    #Optimizer(pw,[90]),
                    #Arima(pw), 
                    #SameValue(pw),
                    #LastNDelta(pw, 30), 
                    #LastNDelta(pw, 15),
                    #LSTM_predictor(pw),
                    #Mean_predictor(pw),
                    #Error_predictor(pw)
                    ]

    if 'result' in item:
         calculated_predictors = list(map(lambda x: x['predictor'], item['result']))
         #predictors = list(filter(lambda x: x.name not in calculated_predictors, predictors))
    logging.info("Predictors: {}".format(list(map(lambda x: x.name, predictors))))
    # Make prediction for every predictor
    success = list(map(lambda predictor: predictor.calc_predictions(error_times), predictors))
    # if a predictor is not successfull return null, because it can not be compared to the others
    if not all(success):
        return None

    # Calculate the prediction errors for each predictor
    errors = calculate_errors(predictors, pw)
    
    # Get arima order
    
    arm = list(filter(lambda x: 'Arima Predictor' in x.name, predictors))
    if len(arm):
        order = arm[0].order
    else:
        order = None

    # Get features created by optimizer
    opt = list(filter(lambda x: 'Optimizer' in x.name, predictors))
    # if len(opt):
    #     features = opt[0].get_features()
    # else:
    #     features = None
    features = None
    
    if pw.plot:

        graphs = list(map(lambda predictor: predictor.get_graph(), predictors))
        plot_graphs(pw, graphs, errors, predictors)

    return errors, order, features


def plot_graphs(pw: PredictionWindow, graphs, errors, predictors: [Predictor]):
    # set figure size
    fig = plt.figure(figsize = (20, 16))
    gs = gridspec.GridSpec(3, 1, height_ratios = [3, 1, 1])
    subplot_iterator = iter(gs)

    # Set Title to day of week and starting time
    # fig.suptitle(get_header(pw))
    #fig.suptitle("Optimized Carb Events")
    # BLOOD GLUCOSE PREDICTION
    plot_bg_prediction(plt.subplot(next(subplot_iterator)), pw, graphs)

    # EVENTS ORIGINAL
    plot_events(plt.subplot(next(subplot_iterator)), pw)

    # EVENTS OPTIMIZED
    plot_events_optimized(plt.subplot(next(subplot_iterator)), pw, predictors)

    # IOB / COB
    # plot_iob_cob(plt.subplot(next(subplot_iterator)), pw, predictors)

    # # PREDICTION PLOT
    # plot_graph_prediction(plt.subplot(next(subplot_iterator)), pw, graphs)

    # # ERRORS
    # plot_errors(plt.subplot(next(subplot_iterator)), pw, errors)

    # SAVE PLOT TO FILE
    plt.savefig(path + "results/plots/graphs-" + pw.startTime.strftime('%Y-%m-%d-%H-%M') + ".png", dpi = 300)
    plt.close()


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
    plt.rc('font', size=14)


def calculate_errors(predictors: [], pw: PredictionWindow) -> []:
    errors = []
    for predictor in predictors:
        error = {'predictor': predictor.name,
                 'errors': predictor.prediction_values - pw.real_values}
        errors.append(error)
    return errors


def plotLegend():
    # Plot Legend
    #plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.tight_layout(pad = 6)
    plt.legend(loc='upper right')


def get_header(pw: PredictionWindow):
    return "{}, {}".format(pw.startTime.day, pw.startTime.strftime('%H:%M'))


def plot_bg_prediction(ax, pw: PredictionWindow, graphs: []):
    setupPlot(ax, pw, 400, 50)
    plt.title("Blood Glucose Level Prediction")
    # Plot real blood glucose readings
    colors_iter = iter([colors[1], colors[7], colors[6]])
    plt.plot(pw.cgmY, color=next(colors_iter), alpha = 0.8, label = "Real BG")
    for graph in graphs:
        plt.plot(graph['values'], color=next(colors_iter), alpha = 0.8, label = 'Simulated BG with optimized carbs')


    values, iob, cob = predict.calculateBG(pw.events, pw.userData)
    plt.plot(values[5], color=next(colors_iter), alpha = 0.8, label = 'Simulated BG with recorded carbs')
    plotLegend()
    plt.ylabel("Blood Glucose level [mg/dL]")
    plt.xlabel("Time [minutes]")
    # plot second x-Axis
    # ax2 = ax.twiny()

    # ax2.xaxis.set_ticks_position('top')  # set the position of the second x-axis to bottom
    # ax2.xaxis.set_label_position('top')  # set the position of the second x-axis to bottom
    # ax2.spines['top'].set_position(('outward', 36))
    # labels = list(map(lambda p: p.strftime('%H:%M'), pd.period_range(pw.startTime, pw.endTime, freq='h')))
    # ax2.set_xticks(ax.get_xticks())
    # ax2.set_xticklabels(labels)
    # ax2.set_xlim(ax.get_xlim())


def plot_events(ax, pw: PredictionWindow):
    setupPlot(ax, pw, 50, 10)
    plt.title("Recorded insulin and carb values")
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
        plt.bar(carbValues.time, carbValues.grams, 5, color=colors[3], alpha = 0.8, label = "Carb event")
    if not bolusValues.empty:
        plt.bar(bolusValues.time, bolusValues.units, 5, color=colors[9], alpha = 0.8, label = "Bolus event")
    plotLegend()
    plt.ylabel("Carbs [gram], Bolus [insulin units]")
    plt.xlabel("Time [minutes]")


def plot_events_optimized(ax, pw, predictors):
    opt = None
    for predictor in predictors:
        if isinstance(predictor, Optimizer):
            opt = predictor
            break
    if not opt:
        return
    setupPlot(ax, pw, 15, 2)
    plt.title("Optimized Carb Values")
    basalValues = opt.all_events[opt.all_events.etype == 'tempbasal']
    carbValues = opt.all_events[opt.all_events.etype == 'carb']
    bolusValues = opt.all_events[opt.all_events.etype == 'bolus']
    # Plot Events
    # logger.debug(basalValues.values[0])
    if not basalValues.empty:
        # plt.bar(basalValues.time, basalValues.dbdt, 5, alpha = 0.8, label = "basal event")
        pass
    # logger.debug(carbValues)
    if not carbValues.empty:
        ctypes = carbValues.ctype.unique()
        colors_iter = iter([colors[5], colors[8], colors[10]])

        positions = iter([-4, 0, 4])
        for ctype in ctypes:
            events = carbValues[carbValues.ctype == ctype]
            color = next(colors_iter)
            position = next(positions)
            plt.bar(events.time + position, events.grams, 3, alpha = 0.8,
                    label = "Carb duration: {}".format(ctype), color = color)
    if not bolusValues.empty:
        # plt.bar(bolusValues.time, bolusValues.units, 5, alpha = 0.8, label = "bolus event")
        pass
    plotLegend()
    plt.ylabel("Bolus [insulin units]")
    plt.xlabel("Time [minutes]")


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
    positions = iter(np.linspace(-5, 5, num=len(errors), endpoint=True))
    for error in errors:
        plt.bar(error['errors'].index + next(positions), error['errors'].tolist(), 10 / len(errors), alpha = 0.5,
                label = error['predictor'])
    plotLegend()


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



