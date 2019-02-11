import logging
import os
from datetime import datetime
import extractor
import numpy as np
import pandas

import optimizer
import predict
from matplotlib import gridspec
from matplotlib import pyplot as plt

from Classes import PredictionWindow

logger = logging.getLogger(__name__)

timeFormat = "%d.%m.%y,%H:%M%z"
timeZone = "+0100"


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


def getPredictionVals(cgmX, cgmY, index,  simData):
    zero = simData[int(cgmX[index])]
    start = cgmY[index]
    res = []
    for i in range(int(cgmX[index]), len(simData)):
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

    # Check if there is data
    if pw.cgmY is None:
        return None
    # Check if there is data around the 10h mark
    logger.debug(type(pw.cgmY))
    if not (pw.cgmY.index[-1] >= (pw.userData.simlength - 1) * 60):
        logger.info(pw.cgmY)
        logger.warning("not able to predict")
        return None

    # Set initial Blood Glucose Level
        pw.userData.bginitial = pw.cgmY[0]

    # Get Train Datapoint
    pw.train_value = pw.data.at[float(pw.userData.simlength * 60 - pw.userData.predictionlength), 'cgmValue']

    # Get last Datapoint
    pw.lastValue = pw.data.at[float(pw.userData.simlength * 60), 'cgmValue']
    logger.debug(pw.train_value)
    logger.debug(pw.lastValue)

    # Get Events for Prediction
    events = extractor.getEvents(pw.data)
    if events.empty:
        logger.warning("No events found")
        return None
    # convert to dataframe
    events = pandas.DataFrame([vars(e) for e in events], index=events.index)
    # Only select events which are not in the prediction timeframe
    pw.events = events[events.index < pw.userData.simlength * 60 - pw.userData.predictionlength]

    # check if events are in prediction frame, if yes, return none because we want only data without these events
    events_in_prediction = events[events.index >= pw.userData.simlength * 60 - pw.userData.predictionlength]
    events_in_prediction = events_in_prediction[events_in_prediction['etype'] != 'tempbasal']
    if not events_in_prediction.empty:
        return None

    # Run Prediction
    if pw.plot:
        sim_bg = predict.calculateBG(pw.events, pw.userData)
        # Get prediction Value for last train value
        prediction_last_train = np.array([sim_bg[0][pw.time_last_train], sim_bg[5][pw.time_last_train]])
        #logger.debug("prediction train " + str(prediction_last_train))
        # Get prediction Value for last value
        prediction_last_value = np.array([sim_bg[0][pw.time_last_value], sim_bg[5][pw.time_last_value]])
        #logger.debug("prediction value " + str(prediction_last_value))
        # get prediction with optimized parameters
        prediction_optimized, optimized_curve = optimizer.optimize(pw)
        #logger.info("optimizer prediction " + str(prediction_optimized))
    else:
         # Get prediction Value for last train value
        prediction_last_train = np.array(predict.calculateBGAt2(pw.userData.simlength * 60 - pw.userData.predictionlength, pw.events, pw.userData))
        #logger.debug("prediction train " + str(prediction_last_train))
        # Get prediction Value for last value
        prediction_last_value = np.array(predict.calculateBGAt2(pw.userData.simlength * 60, pw.events, pw.userData))
        #logger.debug("prediction value " + str(prediction_last_value))
        prediction_optimized = optimizer.optimize(pw)
        #logger.info("optimizer prediction " + str(prediction_optimized))
        
 

    
    # Get Delta between train and last value
    prediction_delta = prediction_last_value - prediction_last_train
    #logger.debug("delta " + str(prediction_delta))
    # add on last Train value
    prediction = np.add(pw.train_value, prediction_delta)
    #logger.debug("prediction " + str(prediction))
    # add same value prediction
    prediction = np.append(prediction, pw.train_value)
    #logger.debug("prediction " + str(prediction))
    # add last 30 min prediction
    last30delta = pw.train_value - pw.cgmY[pw.userData.simlength * 60 - pw.userData.predictionlength - 30]
    prediction30delta = last30delta * pw.userData.predictionlength / 30     # convert string to float and extend it to prediction length
    prediction30 = pw.train_value + prediction30delta
    prediction = np.append(prediction, prediction30)
    pw.prediction = np.append(prediction, prediction_optimized)
    #logger.debug("prediction " + str(prediction))
    # calculate error
    pw.errors = np.subtract(pw.lastValue, pw.prediction)
    #logger.debug("errors " + str(errors))

    if pw.plot:
        # get values for prediction timeframe
        prediction_vals = getPredictionVals(pw.cgmX, pw.cgmY, pw.index_last_train, sim_bg[0])
        prediction_vals_adv = getPredictionVals(pw.cgmX, pw.cgmY, pw.index_last_train, sim_bg[5])

        # Plot

        basalValues = df[df.etype == 'tempbasal']
        carbValues = df[df.etype == 'carb']
        bolusValues = df[df.etype == 'bolus']

        fig = plt.figure(figsize=(10, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        #fig, ax = plt.subplots()

        ax = plt.subplot(gs[0])
        plt.xlim(0, pw.userData.simlength * 60 + 1)
        plt.ylim(0, 400)
        plt.grid(color="#cfd8dc")
        # Major ticks every 20, minor ticks every 5
        major_ticks_x = np.arange(0, pw.userData.simlength * 60 + 1, 60)
        minor_ticks_x = np.arange(0, pw.userData.simlength * 60 + 1, 15)
        major_ticks_y = np.arange(0, 401, 50)
        #minor_ticks_x = np.arange(0, 400, 15)

        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)

        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
        plt.box(False)

        # And a corresponding grid
        #ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)


        # Plot Line when prediction starts
        plt.axvline(x=(pw.userData.simlength - 1) * 60, color="black")
        # Plot real blood glucose readings
        plt.plot(pw.cgmX, pw.cgmY, "#263238", alpha=0.8, label="real BG")
        # Plot sim results
        plt.plot(sim_bg[3], sim_bg[0], "#b71c1c", alpha=0.5, label="sim BG")
        plt.plot(range(int(pw.cgmX[pw.index_last_train]), len(sim_bg[3])), prediction_vals, "#b71c1c", alpha=0.8, label="SIM BG Pred")
        plt.plot(sim_bg[3], sim_bg[5], "#4527a0", alpha=0.5, label="sim BG ADV")
        plt.plot(range(int(pw.cgmX[pw.index_last_train]), len(sim_bg[3])), prediction_vals_adv, "#4527a0", alpha=0.8, label="SIM BG Pred ADV")
        # Same value prediction
        plt.axhline(y=prediction_vals[0], xmin=(pw.userData.simlength - pw.userData.predictionlength / 60) / pw.userData.simlength, alpha=0.8, label="Same Value Prediction")
        # last 30 prediction value
        plt.plot([(pw.userData.simlength - 1) * 60, pw.userData.simlength * 60], [pw.train_value, prediction30], "#388E3C", alpha=0.8, label="Last 30 Prediction")
        # optimized prediction
        plt.plot(optimized_curve, alpha=0.8, label="optimized curve")
        # Plot Legend
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(pad=6)

        # Plot Insulin and Carb graph
        #plt.plot(data[3], data[1], "#64dd17", alpha=0.5, label="sim BGC")
        #plt.plot(data[3], data[2], "#d50000", alpha=0.5, label="sim BGI")
        #plt.plot(data[3], data[4], "#aa00ff", alpha=0.5, label="sim BGI ADV")

        ax = plt.subplot(gs[1])

        major_ticks_x = np.arange(0, pw.userData.simlength * 60 + 1, 60)
        minor_ticks_x = np.arange(0, pw.userData.simlength * 60 + 1, 15)
        major_ticks_y = np.arange(0, 11, 2)
        # minor_ticks_x = np.arange(0, 400, 15)

        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
        plt.box(False)

        # Plot Events
        plt.xlim(0, pw.userData.simlength * 60 + 1)
        plt.ylim(0, 10)
        plt.grid(color="#cfd8dc")
        logger.debug(basalValues.values[0])
        if(len(basalValues) > 0):
            plt.bar(basalValues.time, basalValues.dbdt, 5, alpha=0.8, label="basal event (not used)")
        logger.debug(carbValues)
        if len(carbValues) > 0:
            plt.bar(carbValues.time, carbValues.grams, 5, alpha=0.8, label="carb event")
        if len(bolusValues) > 0:
            plt.bar(bolusValues.time, bolusValues.units, 5, alpha=0.8, label="bolus event")
        #plt.bar(basalValues.time, [0] * len(basalValues), "bo", alpha=0.8, label="basal event (not used)")
        #plt.plot(carbValues.time, [0] * len(carbValues), "go", alpha=0.8, label="carb event")
        #plt.plot(bolusValues.time, [0] * len(bolusValues), "ro", alpha=0.8, label="bolus evnet")
        # Plot Line when prediction starts
        plt.axvline(x=(pw.userData.simlength - 1) * 60, color="black")
        # Plot Legend
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(pad=6)
        plt.subplots_adjust(hspace=0.2)
        # Save plot as svgz (smallest format, able to open with chrome)
        directory = os.path.dirname("/t1d/results/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(directory + '/plots/'):
            os.makedirs(directory + '/plots/')
        plt.savefig("/t1d/results/plots/result-" + pw.startTime.strftime('%Y-%m-%d-%H-%M') + ".png", dpi=150)
        plt.close()

    return pw.errors.tolist()
