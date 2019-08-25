import logging
import coloredlogs
import os
import predictors.predict as diabetes
import pandas as pd
from data.readData import read_data

from Classes import Event, UserData
from matplotlib import pyplot as plt

coloredlogs.install(level = logging.INFO , fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')


path = os.getenv('T1DPATH', '../')

ids = ['27283995', '82923830', '29032313', '1']


def main():
    logging.info("Starting analysis...")

    # load files as dataframes

    datasets = list(map(lambda id: {'id': id, 'data': read_data('data/csv/csv_{}.csv'.format(id))}, ids))

    # get duration
    #durations = list(map(get_durations, datasets))
    list(map(get_time,datasets))
    # get time in normoglycemia

    # get time in hyperglycemia

    # get time in hypoglycemia

    logging.info("DONE")

def get_durations(dataset):
    logging.info("Get duration for {}".format(dataset['id']))
    data = dataset['data']
    logging.info(data.head(5))
    logging.info(data.head(-5))

    logging.info("next:")

def get_time(dataset):
    logging.info("get time for {}".format(dataset['id']))
    data = dataset['data']
    cgmValues : pd.Series = data['cgmValue'].dropna()
    #logging.info(cgmValues.describe())
    total = len(cgmValues)
    logging.info("Total: {}".format(total))
    hypo = cgmValues[cgmValues < 70]
    logging.info("Hypo\t{}\t{}".format(len(hypo), len(hypo)/total))
    norm = cgmValues[(cgmValues >= 70) & (cgmValues <= 140)]
    logging.info("Norm\t{}\t{}".format(len(norm), len(norm)/total))
    hyper = cgmValues[cgmValues > 140]
    logging.info("Hyper\t{}\t{}".format(len(hyper), len(hyper)/total))
    assert len(norm) + len(hypo) + len(hyper) == total



if __name__ == "__main__":
    main()