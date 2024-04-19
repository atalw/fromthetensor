from typing import Tuple
import time
from tqdm import trange
from tinygrad import Tensor, GlobalCounters, TinyJit
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save, load_state_dict
import data
import models.pos2vec as pos2vec_model
import models.siamese as siamese_model
import models.distilled as distilled_model
from tinygrad.helpers import getenv

@TinyJit
def evaluate(x1_test, x2_test, y_test):
  Tensor.training = False
  out = distilled(Tensor.cat(x1_test, x2_test, dim=-1))
  return  ((out.argmax(axis=-1)) == y_test.argmax(axis=-1)).mean()

def deepchess_inference(x1, x2):
  out_one = pos2vec.encode(x1)
  out_two = pos2vec.encode(x2)
  input = Tensor.cat(out_one, out_two, dim=-1)
  return deepchess(input).realize()

@TinyJit
def train_step(x1_train, x2_train, y_train) -> Tuple[Tensor, Tensor]:
  target = deepchess_inference(x1_train, x2_train) # soft target
  with Tensor.train():
    out = distilled(Tensor.cat(x1_train, x2_train, dim=-1))
    loss = out.binary_crossentropy_logits(target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    acc = (out.argmax(axis=-1) == y_train.argmax(axis=-1)).mean()
    return loss.realize(), acc.realize()

if __name__ == "__main__":
  epochs = distilled_model.hyp['epochs']
  learning_rate = distilled_model.hyp['opt']['lr']
  start_epoch = getenv("EPOCH", 0)

  pos2vec = pos2vec_model.Pos2Vec()
  load_state_dict(pos2vec, safe_load("./ckpts/pos2vec_2m.safe"))
  deepchess = siamese_model.Siamese()
  load_state_dict(deepchess, safe_load("./ckpts/deepchess_600k.safe"))
  distilled = distilled_model.Distilled()

  if start_epoch > 0:
    load_state_dict(distilled, safe_load(f"./ckpts/inter/distilled_600k_epoch_{start_epoch-1}.safe"))
    learning_rate *= distilled_model.hyp['opt']['lr_decay']**start_epoch

  opt = optim.Adam(get_parameters(distilled), lr=learning_rate)

  st = time.monotonic()
  for i in (t := trange(start_epoch, epochs)):
    GlobalCounters.reset()
    x1_train, x2_train, y_train = data.load_new_pairs(i)
    cl = time.monotonic()
    loss, acc = train_step(x1_train, x2_train, y_train)
    t.set_description(f"lr: {opt.lr.item():9.9f} loss: {loss.numpy():4.2f} acc: {acc.numpy():5.2f}% {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    st = cl
    del x1_train, x2_train, y_train
    opt.lr = opt.lr * distilled_model.hyp['opt']['lr_decay']
    safe_save(get_state_dict(distilled), f"./ckpts/inter/distilled_600k_epoch_{i}.safe")
  
  x1_test, x2_test, y_test = data.generate_test_set()
  acc = evaluate(x1_test, x2_test, y_test)
  print("test set accuracy is %f" % acc.numpy())

  fn = f"./ckpts/distilled_600k.safe"
  safe_save(get_state_dict(distilled), fn)
  print(f" *** Model saved to {fn} ***")
