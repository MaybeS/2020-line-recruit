from typing import Tuple
from pathlib import Path

import csv
import numpy as np


class Dataset:
    """Dataset
    """
    FILES = ['genome-scores', 'genome-tags', 'links', 'movies', 'ratings', 'tags']

    def __init__(self, path: str):
        self.path = Path(path)
        self.ratings, self.rating_headers = read_csv(str(self.path.joinpath('ratings.csv')), delimiter=',')

    def split_train_test(self, mode: str, axis: int = -1) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Split dataset to train and test set by condition which apply by `axis`.

        :param mode: dataset split mode
        :param axis: dataset split axis
        :return:
            - Train dataset
            - Test dataset
        """
        if mode == 'first':
            train_min, train_max = 1104505203, 1230735592
            test_min, test_max = 1230735600, 1262271552
        elif mode == 'second':
            train_min, train_max = 789652004, 1388502016
            test_min, test_max = 1388502017, 1427784002
        elif mode == 'tiny':
            train_min, train_max = 1104505203, 1104555203
            test_min, test_max = 1230735600, 1230755600
        else:
            raise NotImplementedError(f'{mode} is not implemented')

        return (
            self.ratings[(train_min <= self.ratings[:, axis]) & (self.ratings[:, axis] <= train_max)],
            self.ratings[(test_min <= self.ratings[:, axis]) & (self.ratings[:, axis] <= test_max)]
        )


def read_csv(path: str, **kwargs) \
        -> Tuple[np.ndarray, str]:
    """return csv contents and headers

    :param path: path to file
    :param kwargs: delimiter=',' and skip_header=True is default parameters
    :return:
        - csv values
        - csv headers
    """
    opts = {
        'delimiter': ',',
        'skip_header': True,
    }

    try:
        with open(path) as f:
            reader = csv.reader(f)
            headers = next(reader)
        values = np.genfromtxt(path, **(opts.update(kwargs) or opts))
        return values, ','.join(headers)

    except StopIteration:
        raise Exception('file not readable')


def to_csv(path: str, ary: np.ndarray, **kwargs):
    """write numpy array to file

    :param path: path to file
    :param ary: numpy array
    :param kwargs: delimiter=',' is default parameter
    :return:
    """
    opts = {
        'delimiter': ',',
    }

    np.savetxt(path, ary, **(opts.update(kwargs) or opts))
