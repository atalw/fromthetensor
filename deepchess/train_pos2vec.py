import time
import numpy as np
from tinygrad import Tensor, dtypes, GlobalCounters, TinyJit
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
import models.pos2vec as autoencoder 
import data
import random
from tqdm import trange

@TinyJit
def train_layer_step(x_train, layer) -> Tensor:
  with Tensor.train():
    decoded_out = model(x_train, layer)
    target = x_train if layer == 0 else model.encode(x_train, layer-1)
    loss = decoded_out.binary_crossentropy_logits(target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.realize()

def train_step_wrapper(x_train, layer):
  global st
  GlobalCounters.reset()
  cl = time.monotonic()
  loss = train_layer_step(x_train, layer)
  t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f} {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
  opt.lr = opt.lr * autoencoder.hyp['opt']['lr_decay']
  st = cl

if __name__ == "__main__":
  n = 500_000 # paper uses 2m random positions
  epochs = autoencoder.hyp['epochs']
  model = autoencoder.Pos2Vec()
  learning_rate = autoencoder.hyp['opt']['lr']
  wins, loses = data.load_wins_loses()
  i, j = np.random.randint(low=0, high=wins.shape[0]-n//2), np.random.randint(low=0, high=loses.shape[0]-n//2)
  wins_batch = Tensor(wins[i:i+n//2], dtype=dtypes.float32)
  loses_batch  = Tensor(loses[j:j+n//2], dtype=dtypes.float32)
  del wins, loses
  x_train = Tensor.cat(wins_batch, loses_batch, dim=0)
  x_train = x_train[Tensor.randint(n, high=x_train.shape[0])] # shuffle for good measure

  st = time.monotonic()
  opt = optim.SGD(get_parameters(model.layer0), lr=learning_rate)
  for i in (t := trange(epochs//4)): train_step_wrapper(x_train, 0)

  st = time.monotonic()
  opt = optim.SGD(get_parameters(model.layer1), lr=learning_rate)
  for i in (t := trange(epochs//4)): train_step_wrapper(x_train, 1)

  st = time.monotonic()
  opt = optim.SGD(get_parameters(model.layer2), lr=learning_rate)
  for i in (t := trange(epochs//4)): train_step_wrapper(x_train, 2)

  st = time.monotonic()
  opt = optim.SGD(get_parameters(model.layer3), lr=learning_rate)
  for i in (t := trange(epochs//4)): train_step_wrapper(x_train, 3)
  
  fn = f"./ckpts/pos2vec_1m.safe"
  safe_save(get_state_dict(model), fn)
  print(f" *** Model saved to {fn} ***")