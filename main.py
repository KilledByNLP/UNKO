import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from unko.utils import Model
from unko.activations import ReLU, Sigmoid
from unko.layers import Dense


def main():
    # CONSTANTS
    N_FOLDS = 10 # Cross Validation
    N_EPOCH = 20 # Epoch
    N_HIDDEN_UNITS = 5 * 2 ** 1

    # Generate Data
    inputs = (np.random.randn(5000, 2) * np.pi).tolist()
    outputs = [[np.sin(x) + np.cos(y)] for x, y in inputs]

    # Generate Model
    inputs_folds = np.array_split(inputs, N_FOLDS)
    outputs_folds = np.array_split(outputs, N_FOLDS)

    train_losses = {}
    test_losses = {}
    for e in range(N_EPOCH):
        train_losses[e] = 0.0
        test_losses[e] = 0.0

    for f in range(N_FOLDS):
        model = Model(learning_rate=0.01)
        model.add(Dense(2, N_HIDDEN_UNITS, activation=Sigmoid()))
        model.add(Dense(N_HIDDEN_UNITS, 1))
        # Split train/test
        train_inputs, test_inputs = np.concatenate(
            inputs_folds[:f] + inputs_folds[f + 1:]), inputs_folds[f]
        train_outputs, test_outputs = np.concatenate(
            outputs_folds[:f] + outputs_folds[f + 1:]), outputs_folds[f]

        # Training
        for epoch in range(N_EPOCH):
            print("Epoch #{}".format(epoch))
            for i, o in zip(train_inputs, train_outputs):
                model.fit(np.matrix(i), np.matrix(o))
            train_loss = model.score(train_inputs, train_outputs)
            test_loss = model.score(test_inputs, test_outputs)
            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))
            train_losses[f] += train_loss
            test_losses[f] += test_loss

        fig = plt.figure()
        ax = Axes3D(fig)

        x = np.arange(-1 * np.pi, 1 * np.pi, 0.5)
        y = np.arange(-1 * np.pi, 1 * np.pi, 0.5)
        X, Y = np.meshgrid(x, y)

    df = pd.DataFrame({
        "train": [v / N_FOLDS for v in train_losses.values()],
        "test": [v / N_FOLDS for v in test_losses.values()]
    })
    df.plot(y=['train', 'test'], figsize=(16, 4), alpha=0.5, logy=True)
    plt.savefig("result.png")


if __name__ == '__main__':
    main()
