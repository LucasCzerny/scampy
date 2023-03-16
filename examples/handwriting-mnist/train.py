from mnist import MNIST
import numpy as np

from scampy import Model, Layer, Activation, Loss, Optimizer

print("Loading the dataset...")

mnist_data = MNIST("mnist")

X_train, y_train = mnist_data.load_training()
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = mnist_data.load_testing()
X_test, y_test = np.array(X_test), np.array(y_test)

print("Done")

model = Model.Sequential(
    Layer.Dense(784, 64, Activation.ReLU),
    Layer.Dense(64, 64, Activation.ReLU),
    Layer.Dense(64, 10, Activation.Softmax)
)

model.train(
    X_train, y_train,
    Loss.CategoricalCrossentropy,
    Optimizer.Adam(decay=1e-3),
    epochs=5, batch_size=128
)

model.test(
    X_test, y_test,
    Loss.CategoricalCrossentropy
)

model.save("model_data/version1.model")