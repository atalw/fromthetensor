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
import models.siamese as siamese
from tqdm import trange
from tinygrad.helpers import getenv

# @TinyJit
def train_step(X_train, Y_train) -> Tuple[Tensor, Tensor]:
  with Tensor.train():
    sample = Tensor.randint(BS, high=X_train.shape[0])
    batches = X_train[sample]
    batch_one, batch_two = batches.split(1, dim=1)
    labels = Y_train[sample]

    # according to the paper, pos2vec is part of siamese and weights for pos2vec are updated alongside siamese 
    out_one = pos2vec.encode(batch_one.reshape(-1, 773))
    out_two = pos2vec.encode(batch_two.reshape(-1, 773))
    input = Tensor.cat(out_one, out_two, dim=-1)

    out = model(input)
    loss = out.binary_crossentropy(labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    acc = (out.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
    return loss.realize(), acc.realize()

# @TinyJit
def evaluate(model, X_test, Y_test, BS=128):
  Tensor.training = False
  def numpy_eval(Y_test):
    Y_test_preds_out = np.zeros(list(Y_test.shape))
    for i in trange((len(Y_test)-1)//BS+1):
      x = Tensor(X_test[i*BS:(i+1)*BS])
      batch_one, batch_two = x.split(1, dim=1)

      out_one = pos2vec.encode(batch_one.reshape(-1, 773))
      out_two = pos2vec.encode(batch_two.reshape(-1, 773))
      input = Tensor.cat(out_one, out_two, dim=-1)

      out = model(input)
      Y_test_preds_out[i*BS:(i+1)*BS] = out.numpy()

    return  ((Y_test_preds_out.argmax(axis=-1)) == Y_test.argmax(axis=-1)).mean()

  acc = numpy_eval(Y_test)
  print("test set accuracy is %f" % acc)
  return acc

if __name__ == "__main__":
  BS = 128
  start_epoch = getenv("EPOCH", 0)
  epochs = siamese.hyp['epochs']

  pos2vec = pos2vec_model.Pos2Vec()
  load_state_dict(pos2vec, safe_load("./ckpts/pos2vec_1m.safe"))

  wins, loses = None, None
  data_chunk_size = 200_000
  num_chunks = data.get_data_count()//data_chunk_size
  chunk = start_epoch//(epochs//num_chunks)

  learning_rate = siamese.hyp['opt']['lr']

  model = siamese.Siamese()
  if start_epoch > 0:
    load_state_dict(model, safe_load(f"./ckpts/deepchess_2m_400k_epoch_{start_epoch-1}.safe"))
    wins, loses = data.load_wins_loses(num_chunks-chunk-1, data_chunk_size)
    X_train, Y_train, X_test, Y_test = data.generate_new_pairs(wins, loses)
    chunk += 1
    learning_rate *= siamese.hyp['opt']['lr_decay']**start_epoch

  opt = optim.Adam(get_parameters(model), lr=learning_rate)

  st = time.monotonic()

  for i in (t := trange(start_epoch, epochs)):
    if i == epochs//num_chunks*chunk:
      # go backwards
      wins, loses = data.load_wins_loses(num_chunks-chunk-1, data_chunk_size)
      chunk += 1
    X_train, Y_train, X_test, Y_test = data.generate_new_pairs(wins, loses)
    GlobalCounters.reset()
    cl = time.monotonic()
    loss, acc = train_step(X_train, Y_train)
    t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f} acc: {acc.numpy():5.2f}% {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    opt.lr.assign(opt.lr * siamese.hyp['opt']['lr_decay'])
    st = cl
    del X_train, Y_train
    safe_save(get_state_dict(model), f"./ckpts/deepchess_2m_400k_epoch_{i}.safe")
  
  evaluate(model, X_test.numpy(), Y_test.numpy())

  fn = f"./ckpts/deepchess_2m_400k.safe"
  safe_save(get_state_dict(model), fn)
  print(f" *** Model saved to {fn} ***")
