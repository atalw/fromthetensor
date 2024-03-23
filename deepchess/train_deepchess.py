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
  dat = np.load("data/dataset_1k.npz")
  ratio = 0.8
  X, Y = dat['arr_0'], dat['arr_1']
  X, Y = generate_win_lose_pairs(list(zip(X, Y)))
  X_train, X_test = X[:int(len(X)*ratio)], X[int(len(X)*ratio):]
  Y_train, Y_test = Y[:int(len(Y)*ratio)], Y[int(len(Y)*ratio):]
  X_train = Tensor(X_train, dtype=dtypes.float32)
  X_test = Tensor(X_test, dtype=dtypes.float32)
  Y_train = Tensor(Y_train, dtype=dtypes.float32).reshape([-1, 2])
  Y_test = Tensor(Y_test, dtype=dtypes.float32).reshape([-1, 2])
  print(X_train.shape, Y_train.shape, Y_train[3].numpy())
  return X_train, Y_train, X_test, Y_test

# def binary_cross_entropy(preds, y):
#     loss = (-y*preds.log() - (1-y)*(1-preds).log()).mean()
#     print("shappeeeeessss")
#     print(preds.numpy(), preds.log().numpy())
#     print("loss", loss.numpy())
#     return loss

def generate_win_lose_pairs(XY):
  # input -> [(pos, 0/1), ...]
  wins = list(filter(lambda x: x[1] == 1, XY))
  loses = list(filter(lambda x: x[1] == 0, XY))
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

@TinyJit
def train_step() -> Tuple[Tensor, Tensor]:
  with Tensor.train():
    sample = Tensor.randint(BS, high=X_train.shape[0])
    batches = X_train[sample]
    print(batches.shape)
    batch_one, batch_two = batches.split(1, dim=1)
    labels = Y_train[sample]

    # according to the paper, pos2vec is part of siamese and weights for pos2vec are updated alongside siamese 
    out_one = pos2vec(batch_one)
    out_two = pos2vec(batch_two)

    input = Tensor(np.concatenate((out_one.numpy(), out_two.numpy()), axis=-1))
    print("inputssss", input.shape, input[0].numpy())
    out = model(input).reshape(-1, 2)
    print("outputsss", out[0].numpy(), labels[0].numpy())

    loss = out.binary_crossentropy_logits(labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    acc = (out.argmax(axis=-1) == labels.argmax(axis=-1)).float().mean()
    return loss.realize(), acc.realize()

@TinyJit
def evaluate(model, X_test, Y_test, BS=128):
  Tensor.training = False
  def numpy_eval(Y_test):
    Y_test_preds_out = np.zeros(list(Y_test.shape))
    for i in trange((len(Y_test)-1)//BS+1):
      x = Tensor(X_test[i*BS:(i+1)*BS])
      batch_one, batch_two = x.split(1, dim=2)

      out_one = pos2vec(batch_one)
      out_two = pos2vec(batch_two)
      input = Tensor(np.concatenate((out_one.numpy(), out_two.numpy()), axis=2))
      out = model(input).reshape(-1, 2)

      Y_test_preds_out[i*BS:(i+1)*BS] = out.numpy()
    Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
    return (Y_test.argmax(axis=-1) == Y_test_preds).mean()

  acc = numpy_eval(Y_test)
  print("test set accuracy is %f" % acc)
  return acc

if __name__ == "__main__":
  BS = 128
  pos2vec = pos2vec_model.Pos2Vec()
  load_state_dict(pos2vec, safe_load("./ckpts/pos2vec.safe"))
  model = siamese.Siamese()
  opt = optim.SGD(get_parameters(model), lr=siamese.hyp['opt']['lr'])
  X_train, Y_train, X_test, Y_test = load_data()

  st = time.monotonic()

  for i in (t := trange(siamese.hyp['epochs'])):
    GlobalCounters.reset()
    cl = time.monotonic()
    loss, acc = train_step()
    t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f} acc: {acc.numpy():5.2f}% {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    opt.lr.assign(opt.lr * siamese.hyp['opt']['lr_decay'])
    st = cl
  
  evaluate(model, X_test.numpy(), Y_test.numpy())

  fn = f"./ckpts/deepchess.safe"
  safe_save(get_state_dict(model), fn)
  print(f" *** Model saved to {fn} ***")
