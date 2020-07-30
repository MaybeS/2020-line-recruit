# -*- coding: utf-8 -*-
"""Recommendation (2020 Line recruit test)

This is code for 2020 line recruit test problem B.
Author: Bae Jiun, Maybe
"""

from typing import List

import numpy as np


class Estimator:
    """Base estimator
    """

    def __init__(self):
        """Base estimator
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) \
            -> None:
        """fit estimator to input X, y
        @param: X               train X
        @param: y               train y
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) \
            -> np.ndarray:
        """predict using trained estimator with input X
        @param: X               test X
        """
        raise NotImplementedError
