import time
import numpy as np
from tinygrad import Tensor, dtypes, GlobalCounters, TinyJit, Device
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
import models.pos2vec as autoencoder 
from tqdm import trange

# @TinyJit
def train_step(batch) -> Tensor:
  with Tensor.train():
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
  data = np.load(f"data/dataset_combined.npy", mmap_mode='c')

  st = time.monotonic()

  for i in (t := trange(epochs)):
    GlobalCounters.reset()
    cl = time.monotonic()
    batch = Tensor(data[np.random.choice(data.shape[0], size=BS)], dtype=dtypes.float32, device=Device.DEFAULT)
    loss = train_step(batch)
    t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f} {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    opt.lr = opt.lr * autoencoder.hyp['opt']['lr_decay']
    st = cl
  
  fn = f"./ckpts/pos2vec_2m.safe"
  safe_save(get_state_dict(model), fn)
  print(f" *** Model saved to {fn} ***")