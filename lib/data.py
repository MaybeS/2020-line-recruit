from typing import Tuple
from pathlib import Path

import numpy as np


class Dataset:
    FILES = ['genome-scores', 'genome-tags', 'links', 'movies', 'ratings', 'tags']

    def __init__(self, path: str):
        self.path = Path(path)
        self.ratings = np.genfromtxt(str(self.path.joinpath('ratings.csv')), delimiter=',', skip_header=True)

    def split_train_test(self, axis: int = 3,
                         train_condition: Tuple[int, int] = (1104505203, 1230735592),
                         test_condition: Tuple[int, int] = (1230735600, 1262271552))\
            -> Tuple[np.ndarray, np.ndarray]:

        train_min, train_max = train_condition
        test_min, test_max = test_condition

        return (
            self.ratings[(train_min <= self.ratings[:, axis]) & (self.ratings[:, axis] <= train_max)],
            self.ratings[(test_min <= self.ratings[:, axis]) & (self.ratings[:, axis] <= test_max)]
        )
