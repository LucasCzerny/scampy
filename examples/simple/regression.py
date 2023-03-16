import nnfs
from nnfs.datasets import sine_data

import matplotlib.pyplot as plt

from scampy import Model, Layer, Activation, Loss, Optimizer

nnfs.init()

X, y = sine_data(1000)
X_test, y_test = sine_data(200)

test_model = Model.Sequential(
    Layer.Dense(1, 64, Activation.ReLU),
    Layer.Dense(64, 64, Activation.ReLU),
    Layer.Dense(64, 1, Activation.Linear)
)

test_model.train(
    X, y,
    Loss.MeanSquaredError,
    Optimizer.Adam(learning_rate=0.01, decay=1e-3),
    epochs=1000,
    print_every=100,
    print_first=True
)

test_model.test(X_test, y_test, Loss.MeanSquaredError)

output = test_model.forward(X)

plt.plot(X, output)
plt.show()