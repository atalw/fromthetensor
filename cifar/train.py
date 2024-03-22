# METAL=1 JIT=2 python3 cifar/train.py
# JIT=2 needed to ignore gpu hang graphing bug
import time
import numpy as np
from typing import Tuple
from tinygrad import Tensor, nn, dtypes, Device, TinyJit, GlobalCounters
from tinygrad.nn import optim, Conv2d, BatchNorm2d, Linear
from tinygrad.nn.state import get_parameters
from extra.datasets import fetch_cifar
from extra.training import train, evaluate
from tqdm import trange
from tinygrad.helpers import CI

# https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
# adjusted values for 32x32 input size
class AlexNet:
    def __init__(self):
        self.layers = [
        Conv2d(3, 32, kernel_size=1, bias=False), Tensor.relu,
        lambda x: x.max_pool2d(kernel_size=3, stride=2),
        BatchNorm2d(32),
        Conv2d(32, 64, kernel_size=3, padding=1), Tensor.relu,
        lambda x: x.max_pool2d(kernel_size=3, stride=2),
        BatchNorm2d(64),
        Conv2d(64, 256, kernel_size=3, padding=1), Tensor.relu,
        Conv2d(256, 512, kernel_size=3, padding=1, bias=False), Tensor.relu,
        Conv2d(512, 512, kernel_size=3, padding=1), Tensor.relu,
        lambda x: x.max_pool2d(kernel_size=3, stride=2),
        lambda x: x.flatten(1),
        lambda x: x.dropout(0.5),
        Linear(512*3*3, 4096), Tensor.relu,
        lambda x: x.dropout(0.5),
        Linear(4096, 4096), Tensor.relu,
        Linear(4096, 10), Tensor.relu,
        lambda x: x.log_softmax(axis=1)]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)

if __name__ == "__main__":
    model = AlexNet()
    training_steps = 1000
    BS = 256
    lr = 0.01
    st = time.monotonic()

    X_train, Y_train, X_test, Y_test = fetch_cifar()
    # load data and label into GPU and convert to dtype accordingly
    X_train, X_test = X_train.to(device=Device.DEFAULT).float(), X_test.to(device=Device.DEFAULT).float()
    Y_train, Y_test = Y_train.to(device=Device.DEFAULT), Y_test.to(device=Device.DEFAULT)
    X_train, X_test = X_train.reshape((-1, 3, 32, 32)), X_test.reshape((-1, 3, 32, 32))
    # Y_train, Y_test = Y_train.one_hot(10), Y_test.one_hot(10)
    X_train, Y_train = X_train.cast(dtypes.default_float), Y_train.cast(dtypes.default_float)
    X_test, Y_test = X_test.cast(dtypes.default_float), Y_test.cast(dtypes.default_float)

    @TinyJit
    def train_step() -> Tuple[Tensor, Tensor]:
        with Tensor.train():
            opt.zero_grad()

            sample = Tensor.randint(BS, high=X_train.shape[0])
            batch = X_train[sample]
            labels = Y_train[sample]

            out = model(batch)
            loss = out.sparse_categorical_crossentropy(labels)

            loss.backward()
            opt.step()

            acc = (out.argmax(axis=-1) == labels).mean()
            return loss.realize(), acc.realize()

    for round in range(10):
        opt = optim.SGD(get_parameters(model), lr=lr, momentum=0.9, weight_decay=0.0005)
        if round > 0:
            evaluate(model, X_test.numpy(), Y_test.numpy(), 10)
        for i in (t := trange(training_steps)):            
            GlobalCounters.reset()
            cl = time.monotonic()
            loss, acc = train_step()
            t.set_description(f"loss: {loss.numpy():6.2f} accuracy: {acc.numpy():5.2f} %{GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
            st = cl
        # learning rate divided by 10 3-times in the original paper
        lr *= 0.5
