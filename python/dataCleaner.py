import logging
import os
import readData

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

path = os.getenv('T1DPATH', '../')
filename = path + "data/csv/data-o3.csv"

def main():
    logger.info("data cleaner")
    logger.debug("read data")
    data = readData.read_data(filename)
    logger.debug("data shape: " + str(data.shape))
    logger.debug("data type: " + str(type(data)))


if __name__ == '__main__':
    main()
