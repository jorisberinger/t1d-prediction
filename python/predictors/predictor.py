from abc import ABC, abstractmethod


class Predictor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_predictions(self, error_times: [int]) -> bool:
        pass

    @abstractmethod
    def get_graph(self) -> ({'label': str, 'values': [float]}):
        pass

