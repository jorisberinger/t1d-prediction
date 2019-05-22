import logging
import coloredlogs
import os
import pandas as pd
from data.readData import read_data
from matplotlib import pyplot as plt


coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
data_csv_path_16 = path + 'data/csv/data_16.csv'
data_csv_path_17 = path + 'data/csv/data_17.csv'

def main():
    logging.info("Calculating ISF")
    data_16 = read_data(data_csv_path_16)
    data_17 = read_data(data_csv_path_17)
    data = pd.concat([data_16, data_17])
    isf = eighteen_hundred_rule(data)
    logging.info("ISF calculated by 1800 rule: {}".format(isf))


# Use the 1800 rule to calculate isf
# divide 1800 by the total daily insulin doses
# make boxplot to get a feeling for distribution
def eighteen_hundred_rule(data: pd.DataFrame) -> float:
    # Group data by single days using the date column
    grouped_by_day = data.groupby('date')
    logging.info("Using {} days to average isf".format(len(grouped_by_day)))
    # Count all doses of insulin for each day
    total_insulin_per_day = grouped_by_day.apply(get_total_insulin)
    # divide 1800 by total insulin to get isf estimation
    isf_per_day = 1800 / total_insulin_per_day
    # Create Boxplot to show isf distribution
    isf_per_day.plot.box()
    plt.savefig("{}results/isf-boxplot".format(path))
    # Output distribution to console
    logging.info("isf distribution info: {}".format(isf_per_day.describe()))
    # get isf average
    isf = isf_per_day.mean()
    return isf

def get_total_insulin(data: pd.DataFrame) -> float:
    # get total basal values
    basal_total = data['basalValue'].sum()
    # get total bolus values
    bolus_total = data['bolusValue'].sum()
    # return sum of basal and bolus total values
    return basal_total + bolus_total


if __name__ == "__main__":
    main()