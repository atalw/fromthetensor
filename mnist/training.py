import numpy as np
from PIL import Image
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2d, optim
from extra.datasets import fetch_mnist

from model import LeNet
from extra.augment import augment_img
from extra.training import train, evaluate

def transform(x):
    x = [Image.fromarray(xx).resize((32, 32)) for xx in x]
    x = np.stack([np.asarray(xx) for xx in x], axis=0)
    x = x.reshape(-1, 1, 32, 32)
    return x

X_train, Y_train, X_test, Y_test = fetch_mnist()
X_train = X_train.reshape(-1, 28, 28).astype(np.uint8)
X_test = X_test.reshape(-1, 28, 28).astype(np.uint8)

model = LeNet()
opt = optim.Adam(get_parameters(model), lr=0.002)
epochs = 1000
BS = 32

X_aug = augment_img(X_train)
train(model, X_train, Y_train, opt, epochs, BS=BS, transform=transform) 
print("trained")
accuracy = evaluate(model, X_test, Y_test, BS=BS, transform=transform)
print(accuracy)