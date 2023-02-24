import nnfs
from nnfs.datasets import spiral_data

import matplotlib.pyplot as plt

from scampy import Model, Layer, Activation, Loss, Optimizer, Util

nnfs.init()
X, y = spiral_data(100, 3)

test_model = Model.Sequential(
    Layer.Dense(2, 64, Activation.ReLU, l2_regularizers=(5e-4, 5e-4)),
    Layer.Dense(64, 3, Activation.Softmax)
)

test_model.train(
    X, y,
    Loss.CategoricalCrossentropy,
    Optimizer.Adam(learning_rate=0.002, decay=5e-7),
    epochs=10000,
    print_every=1000,
    print_first=True
)

X_test, y_test = spiral_data(100, 3)
test_model.test(X_test, y_test, Loss.CategoricalCrossentropy)

output = test_model.forward(X_test)
output = Util.to_sparse(output)

plt.scatter(X[:, 0], X[:, 1], c=output, cmap="rainbow")
plt.show()