# -*- coding: utf-8 -*-
"""Recommendation (2020 Line recruit test)

This is code for 2020 line recruit test problem B.
Author: Bae Jiun, Maybe
"""

from typing import Any

import numpy as np

from models import SVD


class Recommender:
    """Recommender
    """

    def __init__(self, algorithm=None, **kwargs):
        self.model = (algorithm or SVD)(**kwargs)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        self.model.fit(X.astype(np.long), y.astype(np.float32))

    def predict(self, X: np.ndarray) \
            -> np.ndarray:
        return self.model.predict(X)
