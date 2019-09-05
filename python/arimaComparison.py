import logging
import os
import random
import time

import coloredlogs
from pmdarima import auto_arima
from matplotlib import gridspec, pyplot as plt
from data.convertData import interpolate_cgm, convert
from data.readData import read_data
import pandas as pd
from ann_visualizer.visualize import ann_viz;

coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data_17_1-6.csv"

test = 60 * 3

time_periods = [ 600, 2400, 3600, 6000]
def prep_csv():
    data = read_data(filename)
    data = convert(data)
    data = interpolate_cgm(data)
    cgm = data['cgmValue']
    cgm.to_csv(path + 'data/csv/cgm_17_1-6.csv')


def calculate_arima(number_of_samples):
    cgm = pd.Series.from_csv(path + 'data/csv/cgm_17_1-6.csv')

    logging.info(len(cgm))
    random_checkpoints = random.sample(range(max(time_periods), len(cgm)), number_of_samples)
    result = pd.DataFrame()
    for time_period in time_periods:
        logging.info("Time period {}".format(time_period))
        all_errors = pd.DataFrame()
        for checkpoint in random_checkpoints:
            logging.info("Checking {}".format(checkpoint))
            train_set = cgm[checkpoint - time_period: checkpoint]
            test_set = cgm[checkpoint: checkpoint + test]
            stepwise_fit = auto_arima(train_set, seasonal = False,
                                      trace = False,
                                      error_action = 'ignore', suppress_warnings = True, stepwise = True)

            preds, conf_int = stepwise_fit.predict(n_periods = test, return_conf_int = True)

            errors = test_set - preds
            errors.index = range(test)
            # logging.info('errors {}'.format(errors))
            all_errors = all_errors.append(errors, ignore_index = True)
        mean = all_errors.abs().mean()
        mean.name = time_period
        result = result.append(mean)

    result.to_csv(path + 'results/arima_comparison-{}.csv'.format(len(random_checkpoints)))
    logging.info("Result {}".format(result))

colors = ['#005AA9','#0083CC','#009D81','#99C000','#C9D400','#FDCA00','#F5A300','#EC6500','#E6001A','#A60084','#721085']
def main():
    logging.info("Start Main")
    # Load File and get cgm Values
    number_of_samples = 100
    #calculate_arima(number_of_samples)
    result = pd.read_csv(path + 'results-old/arima_comparison-{}.csv'.format(number_of_samples), index_col=0)
    result.T.plot(legend=True, colors=[colors[0],colors[3],colors[7],colors[9]])
    plt.xlabel("Time [minutes]")
    plt.ylabel("Absolute Error [mg/dL]")
    plt.title("Comparison of ARIMA models with different training lengths")
    plt.savefig(path + 'results/arima_comparison-plot-new-{}.png'.format(number_of_samples))

    logging.info("Done")


def lstm_mode():


if __name__ == '__main__':
    start_time = time.process_time()
    main()
    logging.info(str(time.process_time() - start_time) + " seconds")
