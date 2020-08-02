import numpy as np


class Criterion:
    """
    """
    def __init__(self):
        pass

    def __call__(self, predictions: np.ndarray, values: np.ndarray) \
            -> float:
        raise NotImplementedError()


class RMSE(Criterion):
    def __call__(self, predictions: np.ndarray, values: np.ndarray) \
            -> float:
        return np.sqrt(np.mean((predictions - values) ** 2))


def get(method: str):
    if method.strip().lower() == 'rmse':
        return RMSE

    raise NotImplementedError(f'{method} is not implemented')
