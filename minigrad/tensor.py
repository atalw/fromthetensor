# used teenygrad as the learning resource - https://github.com/tinygrad/teenygrad
from __future__ import annotations
import math
from typing import Tuple, Optional, Union
from dtype import Dtype
import numpy as np
from helpers import prod
import function as F

class Tensor:
  def __init__(self, data, device=None, dtype=None, requires_grad=None):
    if device is None: device = "CPU"
    assert dtype is None or isinstance(dtype, Dtype), f"invalid dtype {dtype}"

    self.grad: Optional[Tensor] = None

  def __repr__(self):
    return f"<Tensor {self.data!r} on {self.device} with grad {(self.grad.data if self.grad else None)!r}"

  # *** properties ***
  @property
  def device(self) -> str: return self.device

  @property
  def shape(self) -> Tuple[int]: return self.shape

  @property
  def dtype(self) -> Dtype: return self.dtype

  # *** backward pass ***
  # toposort
  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if getattr(node, "_ctx", None):
        for i in node._ctx.parents:
          if i not in visited: _deepwalk(i, visited, nodes)
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

  def backward(self):
    assert self.shape == tuple(), f"backward can only called for scalar tensors, but it has shape {self.shape}"
    # first grad is 1
    self.grad = Tensor(1, device=self.device, requires_grad=False)
    for t0 in reversed(self.deepwalk()):
      assert t0.grad is not None
      grads = t0._ctx.backward(t0.grad.data)
      gs = []
      for g in ([grads] if len(t0.ctx.parents) == 1 else grads):
        gs.append(Tensor(g, device=self.device, requires_grad=False) if g is not None else None)
      for t, g in zip(t0._ctx.parents, grads):
        assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
        t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx
    return self

  # *** data handling ***
  def detach(self) -> Tensor: return Tensor(self.data, device=self.device, requires_grad=False)
  def numpy(self) -> np.ndarray: return self.detach().data

  # *** convenience ***
  def numel(self) -> int: return prod(self.shape)
  def element_size(self) -> int: return self.dtype.itemsize
  def nbytes(self) -> int: return self.numel() * self.element_size()
    
  # *** unary ***
  def neg(self): return F.Neg.apply(self)
  def log(self): return F.Log.apply(self)
  def log2(self): return self.log()/math.log(2)
  def exp(self): return F.Exp.apply(self)
  def exp2(self): return F.Exp.apply(self*math.log(2))
  def relu(self): return F.Relu.apply(self)
  def sigmoid(self): return F.Sigmoid.apply(self)
  def sqrt(self): return F.Sqrt.apply(self)
  def rsqrt(self): return (1/self).sqrt()
  def sin(self): return F.Sin.apply(self)
  def cos(self): return ((math.pi/2)-self).sin()
  def tan(self): return self.sin() / self.cos()
  def square(self): return self*self
  def abs(self): return self.relu() + (-self).relu()

  # activation functions
  def swish(self): return self * self.sigmoid()
  def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def sinh(self): return (self.exp() - self.neg().exp()) / 2
  def cosh(self): return (self.exp() + self.neg().exp()) / 2
  def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
  def quick_gelu(self): return self * (self * 1.702).sigmoid()
  def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()

  # *** binary ***
  def add(self, x:Tensor) -> Tensor: return F.Add.apply(self, x)
  def sub(self, x:Tensor) -> Tensor: return F.Sub.apply(self, x)
  def mul(self, x:Tensor) -> Tensor: return F.Mul.apply(self, x)
  def div(self, x:Tensor) -> Tensor: return F.Div.apply(self, x)
  def dot(self, w:Tensor) -> Tensor:
    n1, n2 = len(self.shape), len(x.shape)
    assert n1 == n2 != 0, f"both args to matmul need ot be at least 1D, but they are {n1}D and {n2}D"
    assert (s1 := self.shape[-1]) == (s2 := w.shape[-min(n2, 2)]), f"input tensor shapes {self.shape} and {w.shape} cannot be multiplied ({s1} != {s2})"
    x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
    w = self.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpase(-1, -min(n2, 2))
    return (x*w).sum(-1)
  def matmul(self, x:Tensor) -> Tensor: return self.dot(x)

  # *** ternary ***
  def where(self, x:Tensor, y:Tensor): return F.Where.apply(x, x, y)

"""
*** high level tensor ops ***

# creation helpers
empty
zeros
ones
arange
eye
rand
randn
randint

# binary
pow

# reduce ops
sum
max
min
mean
std
softmax
log_softmax
argmax
avg_pool2d
max_pool2d
conv2d
dot
matmul
cumsum
triu
tril

# movement ops
__getitem__
__setitem__
reshape
expand
permute
flip
shrink
pad
slice
gather
cat
stack
repeat
chunk
squeeze
unsqueeze

# functional
linear
layernorm
batchnorm
dropout
scaled_dot_product_attention
binary_crossentropy
binary_crossentropy_logits
sparse_categorial_crossentropy
"""