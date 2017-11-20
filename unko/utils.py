import numpy as np
from .metrics import mse

class Model(object):
    def __init__(self, learning_rate=0.01):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.layers = []

    def add(self, layer):
        """
        Add a layer to myself

        Attributes:
            layer (Layer): Layer to be added
        """
        self.layers.append(layer)

    def predict(self, i):
        """
        Predict with trained parameters

        Attributes:
            i (np.matrix): Input
        """
        for layer in self.layers:
            i = layer.forward(i)
        return i

    def fit(self, i, to):
        """
        Train parameters with given data

        Attributes:
            i (np.matrix): Input
            to (np.matrix): Ground Truth
        """
        do = self.predict(i) - to
        for l, layer in enumerate(reversed(self.layers)):
            do = layer.backward(do)
        for l, layer in enumerate(self.layers):
            layer.W -= layer.dW * self.learning_rate
            layer.b -= layer.db * self.learning_rate
        return

    def loss(self, i, to):
        """
        Predict and return loss with given data

        Attributes:
            i (np.matrix): Input
            to (np.matrix): Ground Truth
        """
        po = self.predict(i)
        return np.matrix(mse(to, po))

    def score(self, inputs, outpus):
        """
        Show score

        Attributes:
            inputs (list(list(float))): Input
            outputs (list(list(float))): Ground Truth
        """
        score_sum = np.array([self.loss(np.matrix(inputs[x]), np.matrix(
            outpus[x])) for x, i in enumerate(inputs)])
        return sum(score_sum.flatten()) / len(score_sum)
