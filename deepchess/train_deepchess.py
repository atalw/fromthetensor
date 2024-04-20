from typing import Tuple
import time
from tqdm import trange
from tinygrad import Tensor, GlobalCounters, TinyJit
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save, load_state_dict
import data
import models.pos2vec as pos2vec_model
import models.siamese as siamese
from tinygrad.helpers import getenv

@TinyJit
def evaluate(model, x1_test, x2_test, y_test):
  Tensor.training = False
  input = Tensor.cat(pos2vec.encode(x1_test), pos2vec.encode(x2_test), dim=-1)
  out = model(input)
  acc = (out.argmax(axis=-1) == y_test.argmax(axis=-1)).mean()
  return acc.realize()

@TinyJit
def train_step(x1_train, x2_train, y_train) -> Tuple[Tensor, Tensor]:
  with Tensor.train():
    input = Tensor.cat(pos2vec.encode(x1_train), pos2vec.encode(x2_train), dim=-1)
    out = model(input)
    loss = out.binary_crossentropy_logits(y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    acc = (out.argmax(axis=-1) == y_train.argmax(axis=-1)).mean()
    return loss.realize(), acc.realize()

if __name__ == "__main__":
  start_epoch = getenv("EPOCH", 0)
  epochs = siamese.hyp['epochs']
  pos2vec = pos2vec_model.Pos2Vec()
  load_state_dict(pos2vec, safe_load("./ckpts/pos2vec_400k.safe"))
  learning_rate = siamese.hyp['opt']['lr']
  model = siamese.Siamese()

  if start_epoch > 0:
    load_state_dict(pos2vec, safe_load(f"./ckpts/inter/pos2vec_finetuned_epoch_{start_epoch-1}.safe"))
    load_state_dict(model, safe_load(f"./ckpts/inter/deepchess_epoch_{start_epoch-1}.safe"))
    learning_rate *= siamese.hyp['opt']['lr_decay']**start_epoch

  # pos2vec is finetuned while training siamese 
  opt = optim.Adam(get_parameters(pos2vec) + get_parameters(model), lr=learning_rate)

  st = time.monotonic()
  for i in (t := trange(start_epoch, epochs)):
    GlobalCounters.reset()
    x1_train, x2_train, y_train = data.load_new_pairs(0)
    cl = time.monotonic()
    loss, acc = train_step(x1_train, x2_train, y_train)
    t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f} acc: {acc.numpy():5.2f}% {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
    st = cl
    del x1_train, x2_train, y_train
    opt.lr = opt.lr * siamese.hyp['opt']['lr_decay']
    safe_save(get_state_dict(pos2vec), f"./ckpts/inter/pos2vec_finetuned_epoch_{i}.safe")
    safe_save(get_state_dict(model), f"./ckpts/inter/deepchess_epoch_{i}.safe")
  
  x1_test, x2_test, y_test = data.generate_test_set()
  acc = evaluate(model, x1_test, x2_test, y_test)
  print("test set accuracy is %f" % acc.numpy())

  safe_save(get_state_dict(pos2vec), f"./ckpts/pos2vec_finetuned.safe")
  safe_save(get_state_dict(model), f"./ckpts/deepchess.safe")
  print(f" *** Model saved ***")
