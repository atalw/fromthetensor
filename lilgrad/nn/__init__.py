import math
from typing import cast
from tensor import Tensor
from helpers import prod

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    bound = 1 / math.sqrt(in_features)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None
  
  def __call__(self, x:Tensor):
    return x.linear(self.weight.transpose(), self.bias)


class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    self.kernerl_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    self.stride, self.padding = stride, padding
    self.weight = Tensor.kaiming_uniform(out_channels, in_channels, *self.kernerl_size, a=math.sqrt(5))
    bound = 1 / math.sqrt(cast(int, prod(self.weight.shape[1:])))
    self.bias = Tensor.uniform(out_channels, low=-bound, high=bound) if bias else None
  
  def __call__(self, x:Tensor):
    return x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride)

def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
  return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, bias)

class BatchNorm2d:
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    if affine: self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
    else: self.weight, self.bias = None, None

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x:Tensor):
    if Tensor.training:
      batch_mean = x.mean(axis=(0,2,3))
      y = (x - batch_mean.reshape(shape=[1, -1, 1, 1]))
      batch_var = (y*y).mean(axis=(0,2,3))
      batch_invstd = batch_var.add(self.eps).pow(-0.5)

      if self.track_running_stats:
        self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
        self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * prod(y.shape)/(prod(y.shape)-y.shape[1]) * batch_var.detach())
        self.num_batches_tracked += 1
    else:
      batch_mean = self.running_mean
      batch_invstd = self.running_var.reshape(1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()

    return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd)

"""
nn layers

BatchNorm1d
BatchNorm2d
GroupNorm
InstanceNorm
LayerNorm
LayerNorm2d
Embedding

"""