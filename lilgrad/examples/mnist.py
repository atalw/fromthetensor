# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
import gzip, os, sys
import numpy as np
from tqdm import trange
from lilgrad.tensor import Tensor
from lilgrad.dtype import dtypes
import lilgrad.nn as nn
from lilgrad.helpers import fetch

def _fetch_mnist(file, offset): return Tensor(gzip.open(fetch("https://storage.googleapis.com/cvdf-datasets/mnist/"+file)).read()[offset:])
def mnist():
  return _fetch_mnist("train-images-idx3-ubyte.gz", 0x10).reshape(-1, 1, 28, 28), _fetch_mnist("train-labels-idx1-ubyte.gz", 8), \
         _fetch_mnist("t10k-images-idx3-ubyte.gz", 0x10).reshape(-1, 1, 28, 28), _fetch_mnist("t10k-labels-idx1-ubyte.gz", 8)

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm2d(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm2d(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist()

  model = Model()
  opt = nn.optim.SGD(nn.state.get_parameters(model))

  def train_step() -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      samples = Tensor.randint(512, high=X_train.shape[0])
      # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
      loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
      opt.step()
      return loss

  def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(70)):
    loss = train_step()
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")