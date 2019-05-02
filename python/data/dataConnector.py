import logging
import os

from tinydb import TinyDB, Query, JSONStorage
from tinydb.middlewares import CachingMiddleware


class DataConnector:
    db: TinyDB

    def __init__(self, path):
        self.db = TinyDB(path, storage=CachingMiddleware(JSONStorage))

    def get_db(self) -> TinyDB:
        return self.db
