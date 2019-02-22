import json
import logging

import matplotlib.pyplot as plt
import pandas

logger = logging.getLogger(__name__)


def analyzeFile(filename):
    file = open(filename)
    res = json.load(file)
    createErrorPlots(res)


def createErrorPlots(inputData):
    data = inputData['data']
    count = 0
    for x in data:
        if (x[0] == 0 or x[1] == 0):
            count += 1

    data = pandas.DataFrame(data)
    logger.info("number of results: " + str(len(data)))
    logger.debug("number of zeros: " + str(count))

    plt.plot(data[0], label = "Standard", alpha = 0.6)
    plt.plot(data[1], label = "Adv", alpha = 0.6)
    plt.plot(data[2], label = "Same Value", alpha = 0.6)
    plt.plot(data[3], label = "Last 30 Prediction", alpha = 0.6)
    plt.plot(data[4], label = "Optimized", alpha = 0.6)
    plt.grid(color = "#cfd8dc")
    plt.legend()
    plt.title("error plot data")
    plt.savefig("/t1d/results/errorPlot.png", dpi = 600)

    plt.figure()
    plt.boxplot([data[0], data[1], data[2], data[3], data[4]],
                labels = ["standard", "adv", "same value", "last 30", "optimized"])
    plt.grid(color = "#cfd8dc")
    plt.title("Boxplot Comparison data")
    plt.savefig("/t1d/results/boxplot.png", dpi = 600)


def getSummary(res):
    setNumber = 1  # for debug
    df = pandas.DataFrame(res)
    df = df.apply(abs)
    res_series = pandas.Series(df[0])
    res_series = res_series.apply(abs)
    res_adv_series = pandas.Series(df[1])
    res_same_value = pandas.Series(df[2])
    res_30_value = pandas.Series(df[3])
    res_optimized = pandas.Series(df[4])
    res_arima = pandas.Series(df[5])
    mean = res_series.mean(skipna = True)
    median = res_series.median(skipna = True)
    mean_adv = res_adv_series.mean(skipna = True)
    median_adv = res_adv_series.median(skipna = True)
    mean_same_value = res_same_value.mean(skipna = True)
    median_same_value = res_same_value.median(skipna = True)
    mean_30_value = res_30_value.mean(skipna = True)
    median_30_value = res_30_value.median(skipna = True)
    mean_opt = res_optimized.mean(skipna = True)
    median_opt = res_optimized.median(skipna = True)
    mean_arima = res_arima.mean(skipna = True)
    median_arima = res_arima.median(skipna = True)

    jsonobject = {"mean": float(mean), "mean_adv": float(mean_adv), "mean_same_value": float(mean_same_value),
                  "mean_30_value": float(mean_30_value), "mean_optimized": float(mean_opt),
                  "mean_arima": float(mean_arima),
                  "median": float(median), "median_adv": float(median_adv),
                  "median_same_value": float(median_same_value), "median_30_value": float(median_30_value),
                  "median_optimized": float(median_opt), "median_arima": float(median_arima),
                  "data": res}

    json_output = jsonobject.copy()
    del json_output['data']
    logger.info(json_output)
    file = open("/t1d/results/result-" + str(setNumber) + ".json", 'w')
    file.write(json.dumps(jsonobject))

    return jsonobject


if __name__ == '__main__':
    analyzeFile("result.json")
