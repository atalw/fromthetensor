import math
from tensor import Tensor

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    bound = 1 / math.sqrt(in_features)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None
  
  def __call__(self, x:Tensor):
    return x.linear(self.weight.transpose(), self.bias)

"""
nn layers

Linear
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