import logging
import os
import time

import coloredlogs
from tinydb import TinyDB, JSONStorage
from tinydb.middlewares import CachingMiddleware

from data.dataPrep import add_gradient
from dataPreperation.dataLoader import DataLoader


path = os.getenv('T1DPATH', '../')
data_csv_path = path + 'data/csv/data.csv'

# This method reads in data from a csv file and converts it into sliding windows which are saved into a TinyDB json file. 
def main(db: TinyDB, filepath=data_csv_path):
    logging.info("Start import from csv to database...")
    logging.debug("{} items in db".format(len(db)))
    logging.debug("Init data loader")
    data_loader = DataLoader(db)
    logging.info("Loading csv file {}".format(os.path.abspath(filepath)))
    data_loader.load(filepath)
    logging.debug("Adding events")
    data_loader.add_events()
    logging.debug("Calculate gradients")
    add_gradient(db)
    logging.debug("Removing invalid samples")
    data_loader.check_valid()
    logging.debug("Done Importing and cleaning")
    data_loader.clean_invalid()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    coloredlogs.install(level = 'INFO',
                        fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')
    start = time.time()
    
    db_path = path + 'data/tinydb/db1.json'
    db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
    main(db)
    end = time.time()
    print("Elapsed Time: {} s".format(end - start))
