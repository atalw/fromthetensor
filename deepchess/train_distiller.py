from typing import Tuple
import time
from tqdm import trange
from tinygrad import Tensor, GlobalCounters, TinyJit, dtypes
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save, load_state_dict
import data
import models.pos2vec as pos2vec_model
import models.siamese as siamese_model
import models.distilled as distilled_model
from tinygrad.helpers import getenv
from pickle import dump, load
import matplotlib.pyplot as plt

@TinyJit
def evaluate(x1_test, x2_test, y_test):
  Tensor.training = False
  input = Tensor.cat(x1_test, x2_test, dim=-1)
  out = distilled(input)
  return ((out.argmax(axis=-1)) == y_test.argmax(axis=-1)).mean()

def deepchess_inference(x1, x2):
  input = Tensor.cat(pos2vec.encode(x1), pos2vec.encode(x2), dim=-1)
  return deepchess(input).realize()

@TinyJit
def train_entire_step(x1_train, x2_train, y_train) -> Tuple[Tensor, Tensor]:
  with Tensor.train():
    input = Tensor.cat(x1_train, x2_train, dim=-1)
    soft_target = deepchess_inference(x1_train, x2_train)
    out = distilled(input)
    loss = out.binary_crossentropy(soft_target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    acc = (out.argmax(axis=-1) == y_train.argmax(axis=-1)).mean() # measure against hard target
    return loss.realize(), acc.realize()

@TinyJit
def train_encode_step(x1_train, x2_train, y_train) -> Tuple[Tensor, Tensor]:
  with Tensor.train():
    input = Tensor.cat(x1_train, x2_train, dim=-1)
    soft_target = Tensor.cat(pos2vec.encode(x1_train), pos2vec.encode(x2_train), dim=-1)
    out = distilled.encode(input)
    loss = (out-soft_target).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    # TODO: is acc correct?
    acc = (out == soft_target).mean()
    return loss.realize(), acc.realize()

def train_step_wrapper(i, fxn):
  global st
  GlobalCounters.reset()
  cl = time.monotonic()
  x1_train, x2_train, y_train = data.load_new_pairs(i)
  loss, acc = fxn(x1_train, x2_train, y_train)
  t.set_description(f"lr: {opt.lr.item():9.9f} loss: {loss.numpy():4.2f} acc: {acc.numpy():5.2f}% {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
  st = cl
  del x1_train, x2_train, y_train
  opt.lr = opt.lr * distilled_model.hyp['opt']['lr_decay']
  return loss, acc

if __name__ == "__main__":
  epochs = distilled_model.hyp['epochs']
  learning_rate = distilled_model.hyp['opt']['lr']
  start_epoch = getenv("EPOCH", 0)

  pos2vec = pos2vec_model.Pos2Vec()
  load_state_dict(pos2vec, safe_load("./ckpts/pos2vec_finetuned.safe"))
  deepchess = siamese_model.Siamese()
  load_state_dict(deepchess, safe_load("./ckpts/deepchess.safe"))
  distilled = distilled_model.Distilled()

  # mimic feature extraction first, then train entire network
  if start_epoch == 0:
    opt = optim.Adam(get_parameters(distilled.layers1), lr=learning_rate)
    st = time.monotonic()
    for i in (t := trange(epochs)): train_step_wrapper(i, train_encode_step)
  else:
    load_state_dict(distilled, safe_load(f"./ckpts/inter/distilled_epoch_{start_epoch-1}.safe"))
    learning_rate *= distilled_model.hyp['opt']['lr_decay']**start_epoch

  opt = optim.Adam(get_parameters(distilled), lr=learning_rate)
  history = {}
  st = time.monotonic()
  for i in (t := trange(start_epoch, epochs)):
    loss, acc = train_step_wrapper(i, train_entire_step)
    history[i], history[i]['loss'], history[i]['acc'] = {}, loss.numpy(), acc.numpy()
    safe_save(get_state_dict(distilled), f"./ckpts/inter/distilled_epoch_{i}.safe")
    with open('distilled_history.pkl', 'wb') as f: dump(history, f)
  
  x1_test, x2_test, y_test = data.generate_test_set()
  acc = evaluate(x1_test, x2_test, y_test)
  print("test set accuracy is %f" % acc.numpy())

  fn = f"./ckpts/distilled.safe"
  safe_save(get_state_dict(distilled), fn)
  print(f" *** Model saved to {fn} ***")

  history = load(open('distilled_history.pkl', 'rb'))
  losses = {k:v['loss'] for k,v in history.items()}
  accs = {k:v['acc'] for k,v in history.items()}
  plt.plot(losses.keys(), losses.values(), label='loss')
  plt.plot(accs.keys(), accs.values(), label='acc')
  plt.show()