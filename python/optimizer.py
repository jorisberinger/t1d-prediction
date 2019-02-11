import json
import logging
from datetime import timedelta
import time
import pandas
from scipy.optimize import minimize
import check
import extractor
from data import readData
import rolling
import predict
from Classes import UserData, Event, PredictionWindow
from matplotlib import pyplot as plt, gridspec
import numpy as np
import cProfile
import os

logger = logging.getLogger(__name__)

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data.csv"
resultPath = path + "results/"

# use example User data
udata = UserData(bginitial=100.0, cratio=5, idur=4, inputeeffect=None, sensf=41, simlength=11, predictionlength=60,
                     stats=None)

# Set True if minimizer should get profiled
profile = False
# Set carb duration
carb_duration = 90

# get vectorized forms of function for predicter
vec_get_insulin = np.vectorize(predict.calculateBIAt, otypes=[float], excluded=[0, 1, 2])
vec_get_carb = np.vectorize(predict.calculateCarbAt, otypes=[float], excluded=[1, 2])


def optimize(pw: PredictionWindow) -> int:

    # set error time points
    t = np.arange(0,pw.userData.simlength * 60 - pw.userData.predictionlength , 15)
    logger.debug(t)
    t_index = 0
    real_values = np.array(pw.cgmY.loc[t])
    #logger.info("real values " + str(real_values))

    # set number of parameters
    numberOfParameter = 40
    # set inital guess to 0 for all input parameters
    x0 = np.array([1] * numberOfParameter)
    # set all bounds to 0 - 1
    ub = 20
    lb = 0
    bounds = np.array([(lb, ub)] * numberOfParameter)
    #logger.debug("bounds " + str(bounds))
    #logger.info(str(numberOfParameter) + " parameters set")

    #logger.info("check error at: " + str(t))
    # enable profiling
    #logger.info("profiling enabled: " + str(profile))
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # get Insulin Values
    insulin_events = pw.events[pw.events.etype == 'bolus']
    insulin_values = np.array([pw.cgmY[0]] * len(t))
    t_ = t[:, np.newaxis]
    varsobject = predict.init_vars(pw.userData.sensf, pw.userData.idur * 60)
    for row in insulin_events.itertuples():
        iv = vec_get_insulin(row, udata, varsobject, t_).flatten()
        insulin_values = insulin_values + iv


    # create Time Matrix
    times = []
    for i in t:
        times.append(predict.vec_cob1(t - i, carb_duration))
    cob_matrix = np.matrix(times)
    #logger.debug(cob_matrix)
    #logger.info(t.shape)

    # get patient coefficient
    patient_coefficient = udata.sensf / udata.cratio

    # multiply patient_coefficient with cob_matrix
    patient_carb_matrix = patient_coefficient * cob_matrix
    #logger.debug(patient_carb_matrix)

    # Minimize predicter function, with inital guess x0 and use bounds to improve speed, and constraint to positive numbers
    values = minimize(predicter, x0, args=(real_values, insulin_values, patient_carb_matrix), method='L-BFGS-B', bounds=bounds,
                      options={'disp': False, 'maxiter': 1000})  # Set maxiter higher if you have Time
    # values = minimize(predicter, x0, args=(t_, insulin_values, patient_carb_matrix), method='TNC', bounds=bounds, options = {'disp': True, 'maxiter': 1000})

    if profile:
        pr.disable()
        pr.dump_stats(resultPath + "optimizer/profile")
    # output values which minimize predictor
    #logger.info("x_min" + str(values.x))
    # make a plot, comparing the real values with the predicter
    #plot(values.x, t)
    # save x_min values
    #with open(resultPath + "optimizer/values-1.json", "w") as file:
    #    file.write(json.dumps(values.x.tolist()))
    #    file.close()

    prediction_value = getPredictionValue(values.x, t, pw)
    #logger.info("prediction value: " + str(prediction_value))
    #logger.info("finished")
    if not pw.plot:
        return prediction_value
    else:
        prediction_curve = getPredictionCurve(values.x, t, pw)
        return prediction_value, prediction_curve


def predicter(inputs, real_values, insulin_values, p_cob):
    # Calculate simulated BG for every real BG value we have. Then calculate the error and sum it up.
    # Update inputs
    carb_values = np.array(np.matmul(inputs, p_cob))
    predictions = carb_values + insulin_values
    error = np.absolute(real_values - predictions.flatten())
    error_sum = error.sum()
    return error_sum

def getPredictionCurve(carb_values: [float], t: [float], predictionWindow: PredictionWindow) -> [float]:
    carbEvents = []
    for i in range(0, len(carb_values)):
        carbEvents.append(Event.createCarb(t[i], carb_values[i] / 12, carb_duration))
    carb_events = pandas.DataFrame([vars(e) for e in carbEvents])
    logger.info("carb Events")
    logger.info(carb_events)
    # remove original carb events from data
    insulin_events = predictionWindow.events[predictionWindow.events.etype != 'carb']
    allEvents = pandas.concat([insulin_events, carb_events])
    logger.info("all events")
    logger.info(allEvents)
    values = predict.calculateBG(allEvents, predictionWindow.userData)
    return values[5]

def getPredictionValue(carb_values: [float], t: [float], predictionWindow: PredictionWindow) -> float:
    carbEvents = []
    for i in range(0, len(carb_values)):
        carbEvents.append(Event.createCarb(t[i], carb_values[i] / 12, carb_duration))
    carb_events = pandas.DataFrame([vars(e) for e in carbEvents])
    # logger.info(carb_events)
    # remove original carb events from data
    insulin_events = predictionWindow.events[predictionWindow.events.etype != 'carb']
    allEvents = pandas.concat([insulin_events, carb_events])
    value = predict.calculateBGAt2(predictionWindow.userData.simlength * 60, allEvents, predictionWindow.userData)
    return value[1]


def optimizeMain():
    logger.info("start optimizing")
    # load data and select time frame
    loadData()
    logger.info("data loaded")
    directory = os.path.dirname(resultPath + "optimizer/")
    if not os.path.exists(directory):
        os.makedirs(directory)


    # set error time points
    t_index = np.arange(0, len(cgmX_train), 3)
    t = cgmX_train[t_index]
    global real_values
    real_values = cgmY_train[t_index]
    logger.info(real_values)

    # set number of parameters
    numberOfParameter = len(t)
    # set inital guess to 0 for all input parameters
    x0 = np.array([1] * numberOfParameter)
    # set all bounds to 0 - 1
    ub = 20
    lb = 0
    bounds = np.array([(lb, ub)] * numberOfParameter)
    logger.debug("bounds " + str(bounds))
    logger.info(str(numberOfParameter) + " parameters set")

    logger.info("check error at: " + str(t))
    # enable profiling
    logger.info("profiling enabled: " + str(profile))
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # get Insulin Values
    insulin_values = np.array([cgmY[0]] * len(t))
    t_ = t[:,np.newaxis]
    varsobject = predict.init_vars(udata.sensf, udata.idur * 60)
    for row in df.itertuples():
        iv = vec_get_insulin(row, udata, varsobject, t_).flatten()
        insulin_values = insulin_values + iv
    plt.plot(t, insulin_values)
    plt.savefig(resultPath + "optimizer/insulin.png", dpi=75)
    plt.close()

    # Create Carb Events
    carbEvents = [];
    for i in range(0,numberOfParameter):
        carbEvents.append(Event.createCarb(i * (udata.simlength - 1) * 60 / numberOfParameter, x0[i], carb_duration))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])

    # create Time Matrix
    times = []
    for i in t:
        times.append(predict.vec_cob1(t-i,carb_duration))
    cob_matrix = np.matrix(times)
    logger.debug(cob_matrix)
    logger.info(t.shape)


    # get patient coefficient
    p_co = udata.sensf / udata.cratio

    # multiply p_co with cob_matrix
    p_cob = p_co * cob_matrix
    logger.debug(p_cob)

    # Minimize predicter function, with inital guess x0 and use bounds to improve speed, and constraint to positive numbers
    values = minimize(predicter, x0, args=(real_values, insulin_values, p_cob), method='L-BFGS-B', bounds=bounds, options = {'disp': True, 'maxiter': 1000})  # Set maxiter higher if you have Time
    #values = minimize(predicter, x0, args=(t_, insulin_values, p_cob), method='TNC', bounds=bounds, options = {'disp': True, 'maxiter': 1000})

    if profile:
        pr.disable()
        pr.dump_stats(resultPath + "optimizer/profile")
    # output values which minimize predictor
    logger.info("x_min" + str(values.x))
    # make a plot, comparing the real values with the predicter
    plot(values.x, t)
    # save x_min values
    with open(resultPath + "optimizer/values-1.json", "w") as file:
        file.write(json.dumps(values.x.tolist()))
        file.close()

    logger.info("finished")





def plot(values, t):
    logger.debug(values)
    carbEvents = []
    for i in range(0, len(values)):
        carbEvents.append(Event.createCarb(t[i], values[i]/12, carb_duration))
    ev = pandas.DataFrame([vars(e) for e in carbEvents])
    # logger.info(ev)
    allEvents = pandas.concat([df, ev])
    # logger.info(allEvents)

    sim = predict.calculateBG(allEvents, udata)
    logger.debug(len(sim))

    # Plot
    basalValues = allEvents[allEvents.etype == 'tempbasal']
    carbValues = allEvents[allEvents.etype == 'carb']
    bolusValues = allEvents[allEvents.etype == 'bolus']

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    # fig, ax = plt.subplots()

    ax = plt.subplot(gs[0])
    plt.xlim(0, udata.simlength * 60 + 1)
    plt.ylim(0, 400)
    plt.grid(color="#cfd8dc")
    # Major ticks every 20, minor ticks every 5
    major_ticks_x = np.arange(0, udata.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, udata.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, 401, 50)
    # minor_ticks_x = np.arange(0, 400, 15)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
    plt.box(False)

    # And a corresponding grid
    # ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    # Plot Line when prediction starts
    plt.axvline(x=(udata.simlength - 1) * 60, color="black")
    # Plot real blood glucose readings
    plt.plot(cgmX, cgmY, "#263238", alpha=0.8, label="real BG")
    # Plot sim results
    plt.plot(sim[5], "g", alpha=0.8, label="sim BG")


    # Plot Legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=6)

    ax = plt.subplot(gs[1])

    major_ticks_x = np.arange(0, udata.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, udata.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, 21, 4)
    # minor_ticks_x = np.arange(0, 400, 15)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
    plt.box(False)

    # Plot Events
    plt.xlim(0, udata.simlength * 60 + 1)
    plt.ylim(0, 20)
    plt.grid(color="#cfd8dc")
    if (len(basalValues) > 0):
        plt.bar(basalValues.time, basalValues.dbdt, 5, alpha=0.8, label="basal event (not used)")
    if len(carbValues) > 0:
        plt.bar(carbValues.time, carbValues.grams, 5, alpha=0.8, label="carb event")
    if len(bolusValues) > 0:
        plt.bar(bolusValues.time, bolusValues.units, 5, alpha=0.8, label="bolus event")
    plt.bar(original_carbs.time, original_carbs.grams, 5, alpha=0.8, label="original Carb")

    # plt.bar(basalValues.time, [0] * len(basalValues), "bo", alpha=0.8, label="basal event (not used)")
    # plt.plot(carbValues.time, [0] * len(carbValues), "go", alpha=0.8, label="carb event")
    # plt.plot(bolusValues.time, [0] * len(bolusValues), "ro", alpha=0.8, label="bolus evnet")
    # Plot Line when prediction starts
    plt.axvline(x=(udata.simlength - 1) * 60, color="black")
    # Plot Legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=6)
    plt.subplots_adjust(hspace=0.2)

    ax = plt.subplot(gs[2])

    major_ticks_x = np.arange(0, udata.simlength * 60 + 1, 60)
    minor_ticks_x = np.arange(0, udata.simlength * 60 + 1, 15)
    major_ticks_y = np.arange(0, 51, 10)
    # minor_ticks_x = np.arange(0, 400, 15)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False)
    plt.box(False)

    # Plot Events
    plt.xlim(0, udata.simlength * 60 + 1)
    plt.ylim(0, 50)
    plt.grid(color="#cfd8dc")

    # plot error values
    err = error.tolist()[0]
    plt.bar(t, err, 5, color="#CC0000", alpha=0.8, label="error")
    plt.axvline(x=(udata.simlength - 1) * 60, color="black")
    # Plot Legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=6)
    plt.subplots_adjust(hspace=0.2)


    plt.savefig(resultPath + "optimizer/result-1.png", dpi=150)


def loadData():
    data = readData.read_data(filename)
    data["datetimeIndex"] = data.apply(lambda row: rolling.convertTime(row.date + ',' + row.time), axis=1)
    data["datetime"] = data["datetimeIndex"]
    data = data.set_index('datetimeIndex')
    startTime = data.index[0]
    subset = data.loc[startTime <= data.index]
    subset = subset.loc[startTime + timedelta(hours=11) > subset.index]
    global subset_train
    subset_train = subset.loc[startTime + timedelta(hours=10) > subset.index]
    global cgmX
    global cgmY
    global cgmX_train
    global cgmY_train
    cgmX, cgmY, cgmP = check.getCgmReading(subset, startTime)
    cgmX_train , cgmY_train, cgmP_train = check.getCgmReading(subset_train, startTime)
    udata.bginitial = cgmY[0]
    # Extract events
    events = extractor.getEvents(subset_train)
    converted = events.apply(lambda event: check.convertTimes(event, startTime))
    global df
    df = pandas.DataFrame([vars(e) for e in converted])

    # Remove Carb events and tempbasal
    global original_carbs
    original_carbs = df[df['etype'] == 'carb']
    df = df[df['etype'] != "carb"]
    df = df[df['etype'] != "tempbasal"]


if __name__ == '__main__':
    start_time = time.process_time()
    optimizeMain()
    logger.info(str(time.process_time() - start_time) + " seconds")
