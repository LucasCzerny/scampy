import nnfs
from nnfs.datasets import spiral_data

import matplotlib.pyplot as plt

from scampy import Model, Layer, Activation, Loss, Optimizer, Util

import numpy as np

nnfs.init()

X, y = spiral_data(200, 2)
y = y.reshape(-1, 1)

X_test, y_test = spiral_data(200, 2)
y_test = y_test.reshape(-1, 1)

test_model = Model.Sequential(
    Layer.Dense(2, 64, Activation.ReLU, l2_regularizers=(5e-4, 5e-4)),
    Layer.Dense(64, 1, Activation.Sigmoid)
)

test_model.train(
    X, y,
    Loss.BinaryCrossentropy,
    Optimizer.Adam(learning_rate=0.001, decay=5e-7),
    epochs=10000,
    print_every=1000,
    print_first=True
)

test_model.test(X_test, y_test, Loss.BinaryCrossentropy)

output = test_model.forward(X_test)
predictions = Util.get_classification(output)

plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="rainbow")
plt.show()