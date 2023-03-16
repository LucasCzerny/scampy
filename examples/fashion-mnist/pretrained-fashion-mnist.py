import numpy as np
from PIL import Image

from scampy import Model

categories = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Bot"
)

def load_image(path):
    img = Image.open(path).convert("L")
    data = img.load()
    data = np.asarray(img)
    data = data.reshape(784)
    return data

model = Model.Sequential.load("model-data/fashion-mnist.model")
image = load_image("fashion-mnist/test.png")
prediction = model.classify(image)[0]

print(categories[prediction])