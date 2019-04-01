import json
import logging
import os

import matplotlib.pyplot as plt
import pandas
from matplotlib import gridspec

from check import setupPlot

logger = logging.getLogger(__name__)

path = os.getenv('T1DPATH', '../')


def analyzeFile(filename):
    file = open(filename)
    res = json.load(file)
    createErrorPlots(res)


def createErrorPlots(means, all_errors):
    fig = plt.figure(figsize = (10, len(all_errors) * 4 + 4))
    gs = gridspec.GridSpec(1 + len(all_errors), 1)
    subplot_iterator = iter(gs)

    plt.subplot(next(subplot_iterator))
    plt.title("MEAN Absolute Error")
    for name, values in means.iterrows():
        values.plot(label=name)
    plt.legend()

    for name, values in all_errors.items():
        plt.subplot(next(subplot_iterator))
        plt.title(name)
        plt.ylim(-400,400)
        values.boxplot()

    plt.savefig(path + "results/errorPlot.png", dpi = 600)


def getSummary(res):
    logger.debug("in get Summary")
    setNumber = 1  # for debug
    all_results = pandas.DataFrame(res)
    summary = pandas.DataFrame()
    all_data = {}
    for i in range(all_results.shape[1]):
        predictor_results = all_results[i]
        logger.debug("Predictor Results {}".format(predictor_results))
        result_matrix = pandas.DataFrame([], columns = predictor_results[0]['errors'].index)
        for result in predictor_results:
            result_matrix = result_matrix.append(result['errors'], ignore_index = True)

        result_mean = abs(result_matrix).mean()
        result_mean.name = predictor_results[0]['predictor']
        summary = summary.append(result_mean)

        result_matrix.name = predictor_results[0]['predictor']
        all_data[result_matrix.name] = result_matrix

    logger.debug("summary {}".format(summary))
    with open(path + "results/result-summary-" + str(setNumber) + ".json", 'w') as file:
        file.write(summary.to_json())
    #with open(path + "results/result-all-" + str(setNumber) + ".json", 'w') as file:
    #    file.write(all_data.to_json())
    return summary, all_data


def plotTable(js):
    print(json.dumps(js, indent=6, sort_keys=True))
    data = pandas.DataFrame(json)
    logger.info(data)


if __name__ == '__main__':
    #analyzeFile("result.json")
    fake = pandas.DataFrame.from_dict({"615":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":5.7418830286,"Arima Predictor":3.4889340636,"Same Value Predictor":20.6365766182,"Last 30 Predictor":30.0393075316,"Last 180 Predictor":23.5604726337},"630":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":17.1583154731,"Arima Predictor":13.3789857299,"Same Value Predictor":39.4005245877,"Last 30 Predictor":84.7310892092,"Last 180 Predictor":45.8222364182},"645":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":27.1973758113,"Arima Predictor":38.429852793,"Same Value Predictor":61.7469421348,"Last 30 Predictor":157.7533909061,"Last 180 Predictor":71.3795098805},"660":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":26.0808473835,"Arima Predictor":76.5243779316,"Same Value Predictor":87.6687635089,"Last 30 Predictor":231.3800983411,"Last 180 Predictor":104.6261893845},"690":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":5.300281884,"Arima Predictor":118.6210275392,"Same Value Predictor":115.4747729892,"Last 30 Predictor":331.0417752374,"Last 180 Predictor":140.9109118025},"720":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":61.550057383,"Arima Predictor":85.2233464262,"Same Value Predictor":77.2334110502,"Last 30 Predictor":364.6560807145,"Last 180 Predictor":111.1482628013},"750":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":150.434464277,"Arima Predictor":25.4038921191,"Same Value Predictor":15.1054529445,"Last 30 Predictor":359.5042086278,"Last 180 Predictor":42.6194362364},"780":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":198.1322300434,"Arima Predictor":24.3529502889,"Same Value Predictor":42.4152818219,"Last 30 Predictor":388.7187226746,"Last 180 Predictor":37.5196733528}})
    createErrorPlots(fake)
