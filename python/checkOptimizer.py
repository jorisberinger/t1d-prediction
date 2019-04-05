import logging
import os

from PredictionWindow import PredictionWindow
import numpy as np
import copy

from predictors.optimizer import Optimizer
from matplotlib import pyplot as plt, gridspec
path = os.getenv('T1DPATH', '../')

def check(pw: PredictionWindow):
    # Build new Prediction window
    prediction_window: PredictionWindow = copy.deepcopy(pw)
    prediction_window.userData.simlength = 3
    prediction_window.userData.predictionlength = 0
    prediction_window.data = pw.data.loc[np.arange(pw.userData.train_length(), pw.userData.simlength * 60 + 1, dtype=float)]
    prediction_window.data.index = np.arange(0.0, len(prediction_window.data))
    prediction_window.set_values(np.array([0]))

    # get Optimizer
    opt = Optimizer(prediction_window, [15])
    carb_values = opt.optimize_only()
    graph_values = opt.get_graph()
    plt.plot(graph_values['values'])
    plt.title(sum(carb_values))
    plt.plot(prediction_window.cgmY)
    # SAVE PLOT TO FILE
    plt.savefig(path + "results/plots/result-opt-" + pw.startTime.strftime('%Y-%m-%d-%H-%M') + ".png", dpi = 300)
    plt.close()
    return sum(carb_values)
