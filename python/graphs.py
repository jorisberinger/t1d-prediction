import logging
import coloredlogs
import os
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt

import predictors.predict as diabetes
from Classes import Event, UserData
coloredlogs.install(level = logging.INFO , fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')
path = os.getenv('T1DPATH', '../')
path = path + 'graphs/'

colors = ['#005AA9','#0083CC','#009D81','#99C000','#C9D400','#FDCA00','#F5A300','#EC6500','#E6001A','#A60084','#721085']

def main():
    logging.info("Start making awesome graphs...")
    diabetes_graphs()
    # time_series_graphs()

def diabetes_graphs():
    logging.info("Diabetes Graphs...")
    carbs_insulin_together()
    # insulin_on_board()
    # carbs_on_board()

def time_series_graphs():
    logging.info("Time Series Graphs...")
    simple_predictor()
    #autocorrelation_plot()

def insulin_on_board():
    logging.info("Insulin on board graph")
    insulin_duration = 3    # Three hours
    time_steps = range(0,3 * 60)
    iob = list(map(lambda x: diabetes.iob(x, insulin_duration), time_steps))
    logging.info(iob)
    plt.plot(iob, linewidth=2, color=colors[3])
    plt.xlabel("Time in minutes")
    plt.xlim(0, 180)
    plt.ylim(0,105)
    plt.ylabel("Remaining Insulin in Percentage")
    plt.title("Insulin on Board")
    plt.savefig(path+'insulin_on_board')
    # plt.show()
    plt.close()

def carbs_on_board():
    logging.info("Insulin on board graph")
    carb_type = 180     # Three hours
    time_steps = range(0, 3 * 60)
    cob = list(map(lambda x: diabetes.cob(x, carb_type) * 100, time_steps))
    plt.plot(cob, linewidth=2, color=colors[7])
    plt.xlabel("Time in minutes")
    plt.xlim(0, 180)
    plt.ylim(0,105)
    plt.ylabel("Carbs on board in Percentage")
    plt.title("Carbohydrates on Board")
    plt.savefig(path+'carbs_on_board')
    plt.close()

def carbs_insulin_together():
    logging.info("Carbs and insulin together")
    events =    [   Event.createCarb(30, 12, 180),
                    Event.createBolus(40, 120),
                    Event.createCarb(200, 5, 180),
                    Event.createBolus(230, 60)
                ]
    events = pd.DataFrame([vars(e) for e in events])
    user_data = UserData(100, 1, 3, None, 1, 7, None, 0)
    [simbg_res, simbgc, simbgi, x, simbgi_adv, simbg_adv], iob_vals , cob_vals = diabetes.calculateBG(events, user_data)
    plt.plot(simbg_res, linewidth=2, color=colors[2], label="Simulated BGL")
    plt.plot(simbgc + user_data.bginitial, linewidth=1, linestyle='dashed', color=colors[7], label="Carbohydrate effects")
    plt.plot(simbgi + user_data.bginitial, linewidth=1, linestyle='dashed', color=colors[3], label="Insulin effects")
    
    logging.info(events.values)
    logging.info(type(events))

    plt.bar(x=[40,230], height=[120,60], width=8, color=colors[3], label="Bolus events")
    # list(map(lambda e: plt.bar(e[0],e[4] / 10, width=5, color=colors[7], label="Carb event"), filter(lambda x: x[1] == 'carb', events.values)))
    # list(map(lambda e: plt.bar(e[0],-e[2] / 10, width=5, color=colors[3], label="Bolus event"), filter(lambda x: x[1] == 'bolus', events.values)))
    plt.bar(events[events['etype'] == 'carb']['time'], events[events['etype'] == 'carb']['grams'] , width=8, color=colors[7], label="Carb events")
    # plt.bar(events[events['etype'] == 'bolus']['time'], events[events['etype'] == 'bolus']['units'] /10, width=8, color=colors[3], label="Bolus events")
    
    plt.xlabel("Time in minutes")
    #plt.xlim(0, 180)
    #plt.ylim(0,105)
    plt.ylabel("Blood glucose level in mg/dL")
    plt.title("Blood glucose level")
    plt.legend()
    plt.savefig(path+'carbs_insulin_together')
    plt.close()


def autocorrelation_plot():
    logging.info("Autocorrelation plot...")
    time = np.arange(0, 1000, 0.1);
    amplitude = np.sin(time)
    cgm = pd.read_csv(path+'auto.csv')
    plot_acf(cgm, lags=np.arange(0,1000,50))

    plt.savefig(path+'autocorrelation')

def simple_predictor():
    logging.info("Simple predictor")
    cgm = pd.read_csv(path+'auto.csv')
    curve = cgm[200:300].values
    p_time = np.arange(70,100,1)
    l_15 = curve[70] + 2 * (curve[70] - curve[55])
    l_30 = curve[70] + curve[70] - curve[40]
    plt.plot(curve[:70], color= colors[0], label="Prediction Variable")
    plt.plot(p_time,curve[70:], linestyle='dashed', color= colors[0])

    plt.plot(p_time, [curve[70]] * 30, color= colors[3], label="Same Value Predictor")
    plt.plot(p_time, [np.mean(curve)] * 30, color= colors[6], label="Mean Value Predictor")
    plt.plot([70,100], [curve[70], l_15], color= colors[8], label="Trend(15) Predictor")
    plt.plot([70,100], [curve[70], l_30], color= colors[10], label="Trend(30) Predictor")
    plt.vlines(70, 170, 210, colors="black", linestyles='dashed')
    plt.title("Simple predictors")
    plt.legend()
    plt.savefig(path+'simple_predictors')


if __name__ == "__main__":
    main()