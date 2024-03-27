import time
import numpy as np
from tinygrad import Tensor, dtypes, GlobalCounters, TinyJit, Device
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from extra.lr_scheduler import OneCycleLR
import models.pos2vec as autoencoder 
from tqdm import trange
import sys

def load_data(chunk_idx, chunk_size):
  print(f"loading chunk {chunk_idx} ({chunk_idx*data_chunk_size})")
  X = X_on_disk[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
  X = Tensor(X, dtype=dtypes.float32)
  return X

# @TinyJit
def train_step(X_train) -> Tensor:
  with Tensor.train():
    sample = Tensor.randint(BS, high=X_train.shape[0])
    batch = X_train[sample]

    decoded_out = model(batch)
    loss = decoded_out.binary_crossentropy_logits(batch)

    opt.zero_grad()
    loss.backward()
    opt.step()
    
    return loss.realize()

if __name__ == "__main__":
  BS = 128
  epochs = autoencoder.hyp['epochs']
  model = autoencoder.Pos2Vec()
  opt = optim.SGD(get_parameters(model), lr=autoencoder.hyp['opt']['lr'])
  X_on_disk = np.load("data/dataset_500k_X.npy", mmap_mode='c')
  data_chunk_size = X_on_disk.shape[0]
  chunk = 0
  num_chunks = X_on_disk.shape[0]//data_chunk_size

  st = time.monotonic()

  for i in (t := trange(epochs)):
    if i == epochs//num_chunks * chunk:
      X_train = load_data(chunk, data_chunk_size)
      chunk += 1
    GlobalCounters.reset()
    cl = time.monotonic()
    loss = train_step(X_train)
    t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f} {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    opt.lr.assign(opt.lr * autoencoder.hyp['opt']['lr_decay'])
    st = cl
  
  fn = f"./ckpts/pos2vec_500k.safe"
  safe_save(get_state_dict(model), fn)
  print(f" *** Model saved to {fn} ***")