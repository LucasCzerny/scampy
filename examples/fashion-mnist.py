import numpy as np
import matplotlib.pyplot as plt

from scampy import Model, Layer, Activation, Loss, Optimizer, Util

def load_dataset(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)

    labels = data[:, 0].astype(int)

    pixels = data[:, 1:]
    pixels = (pixels - 127.5) / 127.5

    images = pixels.reshape(-1, 784)

    return images, labels

print("Loading the dataset...")

X_train, y_train = load_dataset("fashion-mnist/fashion-mnist_train.csv")
X_test, y_test = load_dataset("fashion-mnist/fashion-mnist_test.csv")

print("Done")

model = Model.Sequential(
    Layer.Dense(784, 128, Activation.ReLU),
    Layer.Dense(128, 128, Activation.ReLU),
    Layer.Dense(128, 10, Activation.Softmax)
)

model.train(
    X_train, y_train,
    Loss.CategoricalCrossentropy,
    Optimizer.Adam(decay=5e-5),
    epochs=10,
    batch_size=256
)

model.test(X_test, y_test, Loss.CategoricalCrossentropy)

model.save("model-data/fashion-mnist.model")