import numpy as np


def mse(to, po):
    return np.square(to - po).mean()
