import numpy as np


class Measure:
    def __init__(self):
        pass

    def __call__(self, predictions: np.ndarray, values: np.ndarray) \
            -> float:
        raise NotImplementedError()


class RMSE(Measure):
    def __call__(self, predictions: np.ndarray, values: np.ndarray) \
            -> float:
        return np.sqrt(np.mean((predictions - values) ** 2))
