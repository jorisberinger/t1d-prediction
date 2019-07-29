import json
import logging
import os

import matplotlib.pyplot as plt
import pandas
import numpy as np
from matplotlib import gridspec
from tinydb import TinyDB, where

logger = logging.getLogger(__name__)

path = os.getenv('T1DPATH', '../')


def analyzeFile(filename):
    file = open(filename)
    res = json.load(file)
    createErrorPlots(res)


def createErrorPlots(db):
    logging.info("in all_errors")
    queries = [{'q': where('result').exists() & (where('valid') == True), "fn":"all"},
               {'q': where('result').exists() & (where('valid') == True) & (where('gradient-s') >= 10), "fn": "gradient-over-10"},
               {'q': where('result').exists() & (where('valid') == True) & (where('gradient-s') < 10), "fn": "gradient-under-10"}
               ]
    for query in queries:
        means, data =  getSummary(db, query)
        if len(means):
            plot(means, data, query['fn'])





def plot(means, data, filename):
    fig = plt.figure(figsize = (10, len(data) * 4 + 4))
    gs = gridspec.GridSpec(1 + len(data), 1, height_ratios = [2] + [1] * len(data))
    subplot_iterator = iter(gs)

    plt.subplot(next(subplot_iterator))
    plt.title("MEAN Absolute Error")
    for name, values in means.iterrows():
        values.plot(label = name)
    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.1),
               fancybox = True, shadow = True, ncol = 2)

    for name, values in data.items():
        plt.subplot(next(subplot_iterator))
        plt.title(name)
        plt.ylim(-400, 400)
        values.boxplot()

    plt.savefig("{}results/{}.png".format(path, filename), dpi = 600)

def getResults(db: TinyDB, query):
    with_result = db.search(query['q'])
    #with_result = db.search((where('result').exists()))
    results = list(map(lambda x: x['result'], with_result))
    results = list(filter(lambda x: len(x) == 6, results)) #
    return results

def getSummary(db: TinyDB, query):
    logger.debug("in get Summary")
    res = getResults(db, query)
    res = list(filter(lambda item: len(list(filter(lambda x: 'LSTM' in x['predictor'],item))), res))
    logging.info("Number of results for query {}: {}".format(query, len(res)))
    if len(res) == 0:
        logging.warning("no result")
        return [],[]
    setNumber = 1  # for debug
    #all_results = pandas.DataFrame(res)
    summary = pandas.DataFrame()
    all_data = {}
    # get labels for predictors
    labels = list(map(lambda x: x['predictor'], res[0]))
    # create series for each predicotor
    # join them in dataframe
    all_results = pandas.DataFrame()
    def set_index(x):
        s = pandas.Series(x)
        s.index = np.array([15, 30, 45, 60, 90, 120, 150, 180])
        return s
    for label in labels:
        predictor_errors = pandas.Series(list(map(lambda x: list(filter(lambda y: y['predictor'] == label, x))[0]['errors'], res)), name = label)
        all_results.append(predictor_errors)
        logger.debug("Predictor Results {}".format(predictor_errors))
        result_matrix = predictor_errors.apply(set_index)
        result_mean = abs(result_matrix).mean()
        result_mean.name = label
        summary = summary.append(result_mean)
        

        result_matrix.name = label
        all_data[result_matrix.name] = result_matrix

    logger.debug("summary {}".format(summary))
    with open(path + "results/result-summary-" + query['fn'] + str(setNumber) + ".json", 'w') as file:
        file.write(summary.to_json())
    return summary, all_data


def plotTable(js):
    print(json.dumps(js, indent=6, sort_keys=True))
    data = pandas.DataFrame(json)
    logger.info(data)


if __name__ == '__main__':
    #analyzeFile("result.json")
    fake = pandas.DataFrame.from_dict({"615":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":5.7418830286,"Arima Predictor":3.4889340636,"Same Value Predictor":20.6365766182,"Last 30 Predictor":30.0393075316,"Last 180 Predictor":23.5604726337},"630":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":17.1583154731,"Arima Predictor":13.3789857299,"Same Value Predictor":39.4005245877,"Last 30 Predictor":84.7310892092,"Last 180 Predictor":45.8222364182},"645":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":27.1973758113,"Arima Predictor":38.429852793,"Same Value Predictor":61.7469421348,"Last 30 Predictor":157.7533909061,"Last 180 Predictor":71.3795098805},"660":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":26.0808473835,"Arima Predictor":76.5243779316,"Same Value Predictor":87.6687635089,"Last 30 Predictor":231.3800983411,"Last 180 Predictor":104.6261893845},"690":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":5.300281884,"Arima Predictor":118.6210275392,"Same Value Predictor":115.4747729892,"Last 30 Predictor":331.0417752374,"Last 180 Predictor":140.9109118025},"720":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":61.550057383,"Arima Predictor":85.2233464262,"Same Value Predictor":77.2334110502,"Last 30 Predictor":364.6560807145,"Last 180 Predictor":111.1482628013},"750":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":150.434464277,"Arima Predictor":25.4038921191,"Same Value Predictor":15.1054529445,"Last 30 Predictor":359.5042086278,"Last 180 Predictor":42.6194362364},"780":{"Optimizer Mixed Carb Types Predictor [30, 60, 90, 120, 240]":198.1322300434,"Arima Predictor":24.3529502889,"Same Value Predictor":42.4152818219,"Last 30 Predictor":388.7187226746,"Last 180 Predictor":37.5196733528}})
    createErrorPlots(fake)
