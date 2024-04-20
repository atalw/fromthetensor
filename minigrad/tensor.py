# used teenygrad as the learning resource - https://github.com/tinygrad/teenygrad
from __future__ import annotations
from typing import Tuple, Optional
from dtype import Dtype
import numpy as np
from helpers import prod

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

  # data handling
  def detach(self) -> Tensor: return Tensor(self.data, device=self.device, requires_grad=False)
  def numpy(self) -> np.ndarray: return self.detach().data

  # convenience
  def numel(self) -> int: return prod(self.shape)
  def element_size(self) -> int: return self.dtype.itemsize
  def nbytes(self) -> int: return self.numel() * self.element_size()
    

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

# unary function
neg
log
log2
exp
exp2
relu
sigmoid
sin
sqrt
rqsrt
cos
tan

# unary math
abs

# unary activation
swish
tanh
sinh
gelu
leakyrelu

# binary
add
sub
mul
div
pow
matmul

# ternary
where

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