from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor, dtypes
from tinygrad.helpers import Timing

from tinygrad.helpers import fetch

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from model import IrisModel
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
from tinygrad.helpers import getenv

net = IrisModel()
opt = SGD(get_parameters(net), lr=1e-3)
epochs = 1000
batch_size = 32

iris = load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


with Tensor.train():
    for step in range(epochs):
        sample = np.random.randint(0, X_train.shape[0], size=(batch_size))
        batch = Tensor(X_train[sample], requires_grad=False, dtype=dtypes.float32)
        labels = Tensor(Y_train[sample])

        out = net(batch)
        loss = Tensor.sparse_categorical_crossentropy(out, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # calculate accuracy
        pred = out.argmax(axis=-1)
        acc = (pred == labels).mean()

        if step % 100 == 0:
            print(f"Epoch {step+1}, loss: {loss.numpy()}, acc: {acc.numpy()}")

with Timing("Time: "):
    avg_acc = 0
    for step in range(100):
        sample = np.random.randint(0, X_test.shape[0], size=(batch_size))
        batch = Tensor(X_test[sample], requires_grad=False, dtype=dtypes.float32)
        labels = Tensor(Y_test[sample])

        out = net(batch)

        # calculate accuracy
        pred = out.argmax(axis=-1).numpy()
        avg_acc += (pred == labels.numpy()).mean()

    print(f"Test Accuracy: {avg_acc / 100}")
