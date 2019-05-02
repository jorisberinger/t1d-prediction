import logging
import os
import time

import coloredlogs
from tinydb import TinyDB, JSONStorage
from tinydb.middlewares import CachingMiddleware

from dataPreperation.dataLoader import DataLoader

path = os.getenv('T1DPATH', '../../')
db_path = path + 'data/tinydb/db1.json'


def main():
    logging.info("Starting Main")
    logging.info("Init tinydb from {}".format(os.path.abspath(db_path)))
    db = TinyDB(db_path, storage=CachingMiddleware(JSONStorage))
    logging.info("Init data loader")
    data_loader = DataLoader(db)
    logging.info("Loading csv file")
    data_loader.load(path + 'data/csv/data_17_3-4.csv')
    #data_loader.load(path + 'data/csv/data-o2.csv')
    logging.info("Adding events")
    data_loader.add_events()
    logging.info("Removing invalid samples")
    data_loader.check_valid()
    logging.info("Done")


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    coloredlogs.install(level = 'INFO',
                        fmt = '%(asctime)s %(filename)s[%(lineno)d]:%(funcName)s %(levelname)s %(message)s')
    start = time.time()
    main()
    end = time.time()
    print("Elapsed Time: {} s".format(end - start))
