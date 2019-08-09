import logging
import coloredlogs
import os
from data.readData import read_data
coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
filename = path + 'data/csv/data_16.csv'

def get_mean_from_csv():
    logging.info("Start getting mean")
    # read csv file
    data = read_data(filename)
    # get cgmValues
    cgmValues = data['cgmValue'].dropna()
    logging.info("Loaded {} cgmValues".format(len(cgmValues)))
    logging.debug(cgmValues.describe())

    # calculate mean
    mean = cgmValues.mean()
    logging.info("Mean: {}".format(mean))



    logging.info("END")


if __name__ == "__main__":
    get_mean_from_csv()