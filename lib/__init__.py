import random

import numpy as np


def seed(value: int):
    random.seed(value)
    np.random.seed(value)
