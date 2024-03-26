import time
import random
from typing import Tuple
import numpy as np
from tinygrad import Tensor, dtypes, GlobalCounters, TinyJit, Device
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save, load_state_dict
from extra.training import evaluate
import data
import models.pos2vec as pos2vec_model
import models.siamese as siamese_model
import models.distilled as distilled_model
from tqdm import trange
from tinygrad.helpers import getenv

@TinyJit
def deepchess_inference(X):
    batch_one, batch_two = X.split(1, dim=1)
    out_one = pos2vec.encode(batch_one.reshape(-1, 773))
    out_two = pos2vec.encode(batch_two.reshape(-1, 773))
    input = Tensor.cat(out_one, out_two, dim=-1)
    out = deepchess(input).reshape(-1, 2)
    return out

# @TinyJit
def train_step(X_train, Y_train) -> Tuple[Tensor, Tensor]:
  with Tensor.train():
    sample = Tensor.randint(BS, high=X_train.shape[0])
    batch = X_train[sample]
    labels = Y_train[sample]
    target = deepchess_inference(batch)

    # combining both positions
    input = batch.flatten(start_dim=1)

    out = distilled(input)
    assert out.shape == target.shape

    loss = out.binary_crossentropy(target)

    opt.zero_grad()
    loss.backward()
    opt.step()

    acc = (out.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
    return loss.realize(), acc.realize()

@TinyJit
def evaluate(model, X_test, Y_test, BS=128):
  Tensor.training = False
  def numpy_eval(Y_test):
    Y_test_preds_out = np.zeros(list(Y_test.shape))
    for i in trange((len(Y_test)-1)//BS+1):
      x = Tensor(X_test[i*BS:(i+1)*BS])
      target = deepchess_inference(x)

      input = x.flatten(start_dim=1)
      out = distilled(input)
      Y_test_preds_out[i*BS:(i+1)*BS] = out.numpy()

    return  ((Y_test_preds_out.argmax(axis=-1)) == Y_test.argmax(axis=-1)).mean()

  acc = numpy_eval(Y_test)
  print("test set accuracy is %f" % acc)
  return acc

if __name__ == "__main__":
  BS = 128
  epochs = distilled_model.hyp['epochs']
  wins, loses = None, None
  learning_rate = distilled_model.hyp['opt']['lr']
  start_epoch = getenv("EPOCH", 0)
  data_chunk_size = 50_000
  num_chunks = data.get_data_count()//data_chunk_size
  chunk = start_epoch//(epochs//num_chunks)

  pos2vec = pos2vec_model.Pos2Vec()
  load_state_dict(pos2vec, safe_load("./ckpts/pos2vec_1m.safe"))
  deepchess = siamese_model.Siamese()
  load_state_dict(deepchess, safe_load("./ckpts/deepchess_1m_400k.safe"))
  distilled = distilled_model.Distilled()

  wins, loses = data.load_wins_loses(chunk, data_chunk_size)

  if start_epoch > 0:
    load_state_dict(distilled, safe_load(f"./ckpts/distilled_1m_epoch_{start_epoch-1}.safe"))
    wins, loses = data.load_wins_loses(chunk, data_chunk_size)
    X_train, Y_train, X_test, Y_test = data.generate_new_pairs(wins, loses)
    chunk += 1
    learning_rate *= distilled_model.hyp['opt']['lr_decay']**start_epoch

  opt = optim.Adam(get_parameters(distilled), lr=learning_rate)
  st = time.monotonic()

  for i in (t := trange(start_epoch, epochs)):
    if i == epochs//num_chunks*chunk:
      wins, loses = data.load_wins_loses(chunk, data_chunk_size)
      X_train, Y_train, X_test, Y_test = data.generate_new_pairs(wins, loses)
      chunk += 1
    GlobalCounters.reset()
    cl = time.monotonic()
    loss, acc = train_step(X_train, Y_train)
    t.set_description(f"lr: {opt.lr.item():9.9f} loss: {loss.numpy():4.2f} acc: {acc.numpy():5.2f}% {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    opt.lr.assign(opt.lr * distilled_model.hyp['opt']['lr_decay'])
    st = cl
    safe_save(get_state_dict(distilled), f"./ckpts/distilled_1m_epoch_{i}.safe")
  
  evaluate(distilled, X_test.numpy(), Y_test.numpy())

  fn = f"./ckpts/distilled_1m.safe"
  safe_save(get_state_dict(distilled), fn)
  print(f" *** Model saved to {fn} ***")
