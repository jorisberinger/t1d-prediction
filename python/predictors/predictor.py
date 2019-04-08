from abc import ABC, abstractmethod
import logging

class Predictor(ABC):
    def __init__(self):
        super().__init__()
        #logging.info("init {}".format(self.name))

    @abstractmethod
    def calc_predictions(self, error_times: [int]) -> bool:
        pass

    @abstractmethod
    def get_graph(self) -> ({'label': str, 'values': [float]}):
        pass

