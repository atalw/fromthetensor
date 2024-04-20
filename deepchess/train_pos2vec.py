import time
import numpy as np
from tinygrad import Tensor, GlobalCounters
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
import models.pos2vec as autoencoder 
import data
from tqdm import trange

def train_layer_step(x_train, target, layer) -> Tensor:
  with Tensor.train():
    decoded_out = model(x_train, layer)
    loss = decoded_out.binary_crossentropy_logits(target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.realize()

def train_step_wrapper(x_train, target, layer):
  GlobalCounters.reset()
  loss = train_layer_step(x_train, target, layer)
  t.set_description(f"lr: {opt.lr.item():9.7f} loss: {loss.numpy():4.2f}")
  opt.lr = opt.lr * autoencoder.hyp['opt']['lr_decay']

if __name__ == "__main__":
  n = 400_000 # paper uses 2m random positions
  epochs = autoencoder.hyp['epochs'] // 4
  model = autoencoder.Pos2Vec()
  learning_rate = autoencoder.hyp['opt']['lr']
  x_train = data.load_n_positions(n)

  opt = optim.SGD(get_parameters(model.layer0), lr=learning_rate)
  target = x_train
  for i in (t := trange(epochs)): train_step_wrapper(x_train, target, 0)
  safe_save(get_state_dict(model.layer0), "./ckpts/inter/pos2vec_400k_l0.safe")

  opt = optim.SGD(get_parameters(model.layer1), lr=learning_rate)
  target = model.encode(x_train, 0)
  for i in (t := trange(epochs)): train_step_wrapper(x_train, target, 1)
  safe_save(get_state_dict(model.layer1), "./ckpts/inter/pos2vec_400k_l1.safe")

  opt = optim.SGD(get_parameters(model.layer2), lr=learning_rate)
  target = model.encode(x_train, 1)
  for i in (t := trange(epochs)): train_step_wrapper(x_train, target, 2)
  safe_save(get_state_dict(model.layer2), "./ckpts/inter/pos2vec_400k_l2.safe")

  opt = optim.SGD(get_parameters(model.layer3), lr=learning_rate)
  target = model.encode(x_train, 2)
  for i in (t := trange(epochs)): train_step_wrapper(x_train, target, 3)
  
  safe_save(get_state_dict(model), "./ckpts/pos2vec_400k.safe")
  print(f" *** Model saved ***")