from abc import ABCMeta, abstractmethod

import numpy as np


class Layer(metaclass=ABCMeta):
    """
    Abstract function of layer
    """
    @abstractmethod
    def forward(self, i):
        pass

    @abstractmethod
    def backward(self, do):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size, activation=None):
        """
        Dense Layer

        Attributes:
            input_size (int): Size of input
            output_size (int): Size of ground truth
            activation (Activation): Activation function
        """
        super(Dense, self).__init__()
        self.W = np.random.randn(input_size, output_size) * (1 / np.sqrt(input_size))
        self.b = np.matrix(np.zeros(output_size))
        self.activation = activation
        self.i = None
        self.dW = None
        self.db = None

    def forward(self, i):
        self.i = i
        o = np.dot(i, self.W) + self.b
        if self.activation:
            o = self.activation.forward(o)
        return o

    def backward(self, do):
        if self.activation:
            do = self.activation.backward(do)
        dx = np.dot(do, self.W.T)
        self.dW = np.dot(self.i.T, do)
        self.db = np.matrix(np.sum(do, axis=0))
        return dx
