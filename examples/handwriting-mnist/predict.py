import numpy as np
import matplotlib.pyplot as plt

from scampy import Model, Util

import tkinter as tk
from PIL import Image, ImageChops

model = Model.Sequential.load("model_data/handwriting-mnist.model")

def draw(event):
    x, y = event.x, event.y

    stroke_size = 20
    offset = stroke_size // 2
    canvas.create_rectangle(x - offset, y - offset, x + offset, y + offset, fill="black", outline="black")
    
def predict(canvas, predict_label):
    canvas.postscript(file="image.eps")
    
    img = Image.open("image.eps")
    img = ImageChops.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    
    img = img.convert("L")

    output = model.forward(np.array(img).reshape(-1, 784))
    predict_label["text"] = Util.to_sparse(output)[0]

def clear(canvas):
    canvas.delete("all")

root = tk.Tk()
root.title("Handwriting Recognizer")

draw_text = tk.Label(root, text="The neural net predicted:", font=("Helvetica", 14, "bold"))
draw_text.pack()

predict_label = tk.Label(root, text="Start drawing to get a prediction!", font=("Helvetica", 13))
predict_label.pack()

canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()
canvas.bind("<B1-Motion>", draw)

predict_button = tk.Button(root, text="Predict!", command = lambda: predict(canvas, predict_label))
predict_button.pack()

clear_button = tk.Button(root, text="Clear", command = lambda: clear(canvas))
clear_button.pack()

root.mainloop()
