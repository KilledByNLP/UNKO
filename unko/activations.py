from abc import ABCMeta, abstractmethod

import numpy as np


class Activation(metaclass=ABCMeta):
    """
    Abstract class of activation function
    """

    @abstractmethod
    def forward(self, i):
        pass

    @abstractmethod
    def backward(self, do):
        pass


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, i):
        o = i.copy()
        o[o < 0] = 0.0
        return o

    def backward(self, do):
        dx = do.copy()
        dx[dx < 0] = 0.0
        dx[dx > 0] = 1.0
        return dx


class Sigmoid(Activation):
    def __init__(self, b=1.0):
        super(Sigmoid, self).__init__()
        self.o = None
        self.b = b

    def forward(self, i):
        o = 1 / (1 + np.exp(-i * self.b))
        self.o = o
        return o

    def backward(self, do):
        dx = np.array(do) * np.array(self.b - self.o) * np.array(self.o)
        return dx
