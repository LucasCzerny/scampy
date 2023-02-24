# Label         Description
# T-shirt/top   0
# Trouser       1
# Pullover      2
# Dress         3
# Coat          4
# Sandal        5
# Shirt         6
# Sneaker       7
# Bag           8
# Ankle Bot     9

import numpy as np
from PIL import Image

from scampy import Model

def load_image(path):
    img = Image.open(path).convert("L")
    data = img.load()
    data = np.asarray(img)
    data = data.reshape(784)
    return data

model = Model.Sequential.load("model-data/fashion-mnist.model")
image = load_image("fashion-mnist/test.png")
print(model.classify(image))