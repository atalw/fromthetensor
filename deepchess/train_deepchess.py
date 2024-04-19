import time
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

@TinyJit
def train_step(X1_train, X2_train, Y_train) -> Tuple[Tensor, Tensor]:
  with Tensor.train():
    batch_one = X1_train
    batch_two = X2_train
    labels = Y_train

    # according to the paper, pos2vec is part of siamese and weights for pos2vec are updated alongside siamese 
    out_one = pos2vec.encode(batch_one)
    out_two = pos2vec.encode(batch_two)
    input = Tensor.cat(out_one, out_two, dim=-1)

    out = model(input)
    loss = out.binary_crossentropy_logits(labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    acc = (out.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
    return loss.realize(), acc.realize()

@TinyJit
def evaluate(model, X1_test, X2_test, Y_test):
  Tensor.training = False
  out_one = pos2vec.encode(X1_test)
  out_two = pos2vec.encode(X2_test)
  input = Tensor.cat(out_one, out_two, dim=-1)

  out = model(input)
  acc = (out.argmax(axis=-1) == Y_test.argmax(axis=-1)).mean()
  return acc.realize()

if __name__ == "__main__":
  start_epoch = getenv("EPOCH", 0)
  epochs = siamese.hyp['epochs']

  pos2vec = pos2vec_model.Pos2Vec()
  load_state_dict(pos2vec, safe_load("./ckpts/pos2vec_2m.safe"))

  learning_rate = siamese.hyp['opt']['lr']
  model = siamese.Siamese()

  wins, loses = data.load_wins_loses()

  if start_epoch > 0:
    load_state_dict(model, safe_load(f"./ckpts/deepchess_2m_600k_epoch_{start_epoch-1}.safe"))
    learning_rate *= siamese.hyp['opt']['lr_decay']**start_epoch

  # we aren't generating new (win,loss) pairs each generation so
  # add weight decay for l2 regularization
  opt = optim.Adam(get_parameters(model), lr=learning_rate)

  st = time.monotonic()

  for i in (t := trange(start_epoch, epochs)):
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = data.generate_new_pairs(wins, loses, i >= epochs-1)
    GlobalCounters.reset()
    cl = time.monotonic()
    loss, acc = train_step(X1_train, X2_train, Y_train)
    t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f} acc: {acc.numpy():5.2f}% {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    opt.lr = opt.lr * siamese.hyp['opt']['lr_decay']
    st = cl
    del X1_train, X2_train, Y_train
    safe_save(get_state_dict(model), f"./ckpts/deepchess_2m_600k_epoch_{i}.safe")
  
  x1_test, x2_test, y_test = data.generate_test_set()
  acc = evaluate(model, x1_test, x2_test, y_test)
  print("test set accuracy is %f" % acc.numpy())

  fn = f"./ckpts/deepchess_2m_600k.safe"
  safe_save(get_state_dict(model), fn)
  print(f" *** Model saved to {fn} ***")
