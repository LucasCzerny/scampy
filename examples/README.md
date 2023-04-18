# Examples

## Handwriting MNIST

This project includes a gui application where you can draw digits and let the neural network classify them.

### Demo

<img src="handwriting-mnist/demo.gif"  height="400">

### How to use

Run the `train.py` file to train the model, then run the `predict.py` file to open the gui.

## Fashion MNIST

Neural network trained on the fashion mnist dataset (without convolutional layers).

### Downloading the dataset

In order to run this example, you have to download the [Fashion MNIST dataset from Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and put the csv files into the `dataset` folder.

### How to use

You will first need to run `fashion-mnist.py` to train the neural network on the dataset. The model will then be saved to the `model-data` folder. Then, you can use `pretrained-fashion-mnist.py` to classify an image of your choice. By default, it will load the included `test.png` file from the `dataset` folder.

## Simple examples

This folder includes three files that demonstrate how to to create a Categorical Model (`categorical.py`), do Binary Logistic Regression (`binary.py`) and Regression (`regression.py`) in scampy.

### How to use

`python categorical.py` or
`python binary.py` or
`python regression.py`
