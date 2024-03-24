import time
import random
from typing import Tuple
import numpy as np
from tinygrad import Tensor, dtypes, GlobalCounters, TinyJit, Device
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save, load_state_dict
from extra.training import evaluate
import models.pos2vec as pos2vec_model
import models.siamese as siamese
from tqdm import trange

def load_data():
  print("loading data")
  dat = np.load("data/dataset_100k.npz")
  X, Y = dat['arr_0'], dat['arr_1']
  combined = list(zip(X, Y))
  wins = list(filter(lambda x: x[1] == 1, combined))
  loses = list(filter(lambda x: x[1] == 0, combined))
  return wins, loses

def generate_new_dataset(wins, loses):
  x, y = generate_win_lose_pairs(wins, loses)
  ratio = 0.8
  X_train, X_test = x[:int(len(x)*ratio)], x[int(len(x)*ratio):]
  Y_train, Y_test = y[:int(len(y)*ratio)], y[int(len(y)*ratio):]
  X_train = Tensor(X_train, dtype=dtypes.float32)
  X_test = Tensor(X_test, dtype=dtypes.float32)
  Y_train = Tensor(Y_train, dtype=dtypes.float32).reshape([-1, 2])
  Y_test = Tensor(Y_test, dtype=dtypes.float32).reshape([-1, 2])
  return X_train, Y_train, X_test, Y_test

def generate_win_lose_pairs(wins, loses):
  # input -> [(pos, 0/1), ...]
  random.shuffle(wins)
  random.shuffle(loses)
  x, y = [], []
  for i in range(min(len(wins), len(loses))):
    x1, y1 = wins[i]
    x2, y2 = loses[i]
    if random.random() < 0.5:
      x.append((x1, x2))
      y.append((y1, y2))
    else:
      x.append((x2, x1))
      y.append((y2, y1))
  return x, y

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
    input = Tensor(np.concatenate((out_one.numpy(), out_two.numpy()), axis=-1))

    out = model(input).reshape(-1, 2)

    loss = out.binary_crossentropy(labels)

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
      batch_one, batch_two = x.split(1, dim=1)

      out_one = pos2vec.encode(batch_one.reshape(-1, 773))
      out_two = pos2vec.encode(batch_two.reshape(-1, 773))
      input = Tensor(np.concatenate((out_one.numpy(), out_two.numpy()), axis=-1)).reshape(-1, 200)

      out = model(input)
      Y_test_preds_out[i*BS:(i+1)*BS] = out.numpy()

    return  ((Y_test_preds_out.argmax(axis=-1)) == Y_test.argmax(axis=-1)).mean()

  acc = numpy_eval(Y_test)
  print("test set accuracy is %f" % acc)
  return acc

if __name__ == "__main__":
  BS = 128
  wins, loses = load_data()
  pos2vec = pos2vec_model.Pos2Vec()
  load_state_dict(pos2vec, safe_load("./ckpts/pos2vec.safe"))
  model = siamese.Siamese()
  opt = optim.Adam(get_parameters(model), lr=siamese.hyp['opt']['lr'])

  st = time.monotonic()

  for i in (t := trange(siamese.hyp['epochs'])):
    X_train, Y_train, X_test, Y_test = generate_new_dataset(wins, loses)
    GlobalCounters.reset()
    cl = time.monotonic()
    loss, acc = train_step(X_train, Y_train)
    t.set_description(f"lr: {opt.lr.item():9.9f} loss: {loss.numpy():4.2f} acc: {acc.numpy():5.2f}% {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    opt.lr.assign(opt.lr * siamese.hyp['opt']['lr_decay'])
    st = cl
  
  evaluate(model, X_test.numpy(), Y_test.numpy())

  fn = f"./ckpts/deepchess.safe"
  safe_save(get_state_dict(model), fn)
  print(f" *** Model saved to {fn} ***")
