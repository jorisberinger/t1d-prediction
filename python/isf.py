import logging
import coloredlogs
import os
import pandas as pd
from data.readData import read_data


coloredlogs.install(level = 'INFO', fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')

path = os.getenv('T1DPATH', '../')
data_csv_path = path + 'data/csv/data_17_5.csv'

def main():
    logging.info("Calculating ISF")
    data = read_data(data_csv_path)
    isf = eighteen_hundred_rule(data)
    logging.info("ISF calculated by 1800 rule: {}".format(isf))
def eighteen_hundred_rule(data: pd.DataFrame) -> float:
    # Use the 1800 rule to calculate isf
    # divide 1800 by the total daily insulin doses
    # make boxplot to get a feeling for distribution



    return 0


if __name__ == "__main__":
    main()