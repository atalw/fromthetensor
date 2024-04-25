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


"""
nn layers

Conv1d
Conv2d
BatchNorm1d
BatchNorm2d
GroupNorm
InstanceNorm
LayerNorm
LayerNorm2d
Embedding

"""