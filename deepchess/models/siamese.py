from tinygrad import Tensor, nn
from tinygrad.helpers import prod

hyp = {
  'epochs':               1000,
  'opt': {
    'lr':                 0.01,
    'lr_decay':           0.99
  }
}

# affine = True
class BatchNorm1d:
  def __init__(self, sz:int):
    self.eps, self.track_running_stats, self.momentum = 1e-5, True, 0.1
    self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
    self.track_running_stats = True

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x: Tensor):
    if Tensor.training:
      batch_mean = x.mean(axis=(0, -1))
      y = (x - batch_mean.reshape(shape=[1, -1, 1]))
      batch_var = (y*y).mean(axis=(0, -1))
      batch_invstd = batch_var.add(self.eps).pow(-0.5)

      if self.track_running_stats:
        self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
        self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * prod(y.shape)/(prod(y.shape)-y.shape[1]) * batch_var.detach())
        self.num_batches_tracked += 1
      else:
        batch_mean = self.running_mean
        # NOTE: this can be precomputed for static inference. we expand it here so it fuses
        batch_invstd = self.running_var.reshape(1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()

    return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd)

class Siamese:
  def __init__(self):
    self.layers = [
      nn.Linear(200, 400),
      BatchNorm1d(400),
      lambda x: x.relu(),
      nn.Linear(400, 200),
      BatchNorm1d(200),
      lambda x: x.relu(),
      nn.Linear(200, 100),
      BatchNorm1d(100),
      lambda x: x.relu(),
      nn.Linear(100, 2),
      BatchNorm1d(2),
      # lambda x: x.relu(),
      lambda x: x.sigmoid(),
    ]

  
  def __call__(self, x: Tensor) -> Tensor:
    return x.sequential(self.layers)