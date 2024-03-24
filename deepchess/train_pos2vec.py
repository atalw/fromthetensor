import time
from typing import Tuple
import numpy as np
from tinygrad import Tensor, dtypes, GlobalCounters, TinyJit, Device
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from extra.lr_scheduler import OneCycleLR
import models.pos2vec as autoencoder 
from tqdm import trange

def load_data():
  print("loading data")
  dat = np.load("data/dataset_100k.npz")
  ratio = 0.8
  X, Y = dat['arr_0'], dat['arr_1']
  X_train, X_test = X[:int(len(X)*ratio)], X[int(len(X)*ratio):]
  Y_train, Y_test = Y[:int(len(Y)*ratio)], Y[int(len(Y)*ratio):]
  # X_train = Tensor(X_train, dtype=dtypes.float32).reshape(-1, 1, 1, 773)
  # X_test = Tensor(X_test, dtype=dtypes.float32).reshape(-1, 1, 1, 773)
  X_train = Tensor(X_train, dtype=dtypes.float32)
  X_test = Tensor(X_test, dtype=dtypes.float32)
  Y_train = Tensor(Y_train, dtype=dtypes.float32).unsqueeze(-1)
  Y_test = Tensor(Y_test, dtype=dtypes.float32).unsqueeze(-1)
  return X_train, Y_train, X_test, Y_test

# @TinyJit
def train_step() -> Tensor:
  with Tensor.train():
    sample = Tensor.randint(BS, high=X_train.shape[0])
    batch = X_train[sample]

    recon_batch, _ = model(batch)
    loss = recon_batch.binary_crossentropy(batch)

    opt.zero_grad()
    loss.backward()
    opt.step()
    
    return loss.realize()

if __name__ == "__main__":
  BS = 128
  model = autoencoder.Pos2Vec()
  opt = optim.SGD(get_parameters(model), lr=autoencoder.hyp['opt']['lr'])
  X_train, Y_train, X_test, Y_test = load_data()

  st = time.monotonic()

  for i in (t := trange(autoencoder.hyp['epochs'])):
    GlobalCounters.reset()
    cl = time.monotonic()
    loss = train_step()
    t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f} {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    opt.lr.assign(opt.lr * autoencoder.hyp['opt']['lr_decay'])
    st = cl
  
  fn = f"./ckpts/pos2vec.safe"
  safe_save(get_state_dict(model), fn)
  print(f" *** Model saved to {fn} ***")