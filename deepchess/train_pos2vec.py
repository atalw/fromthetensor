import time
from typing import Tuple
import numpy as np
from tinygrad import Tensor, dtypes, GlobalCounters
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
import models.pos2vec as autoencoder 
from tqdm import trange

def load_data():
  dat = np.load("data/dataset_1k.npz")
  # np.random.shuffle(dat)
  ratio = 0.8
  X, Y = dat['arr_0'], dat['arr_1']
  X_train, X_test = X[:int(len(X)*ratio)], X[int(len(X)*ratio):]
  Y_train, Y_test = Y[:int(len(Y)*ratio)], Y[int(len(Y)*ratio):]
  X_train = Tensor(X_train, requires_grad=False, dtype=dtypes.float32).reshape((-1, 1, 1, 773))
  X_test = Tensor(X_test, dtype=dtypes.float32).reshape((-1, 1, 1, 773))
  Y_train = Tensor(Y_train, dtype=dtypes.float32).unsqueeze(-1)
  Y_test = Tensor(Y_test, dtype=dtypes.float32).unsqueeze(-1)
  print(X_train.shape, Y_train.shape)
  return X_train, Y_train, X_test, Y_test

def train_step() -> Tuple[Tensor, Tensor]:
    with Tensor.train():
        sample = Tensor.randint(BS, high=X_train.shape[0])
        batch = X_train[sample]
        labels = Y_train[sample]

        out = model(batch)
        loss = out.sub(labels).square().mean().sqrt()

        opt.zero_grad()
        loss.backward()
        opt.step()

        acc = (out.argmax(axis=-1) == labels).mean()
        return loss.realize(), acc.realize()

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = load_data()
  model = autoencoder.Pos2Vec()
  opt = optim.Adam(get_parameters(model), lr=autoencoder.hyp['opt']['lr'])
  BS = 128

  st = time.monotonic()

  for i in (t := trange(autoencoder.hyp['epochs'])):
    GlobalCounters.reset()
    cl = time.monotonic()
    loss, acc = train_step()
    t.set_description(f"loss: {loss.numpy():6.2f} accuracy: {acc.numpy():5.2f} %{GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    st = cl