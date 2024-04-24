# used teenygrad as the learning resource - https://github.com/tinygrad/teenygrad
from __future__ import annotations
import math
from typing import Tuple, Optional, Union, Type, List
from dtype import Dtype, dtypes
from device import Device
import numpy as np
from helpers import prod, argfix
import function as F
from buffer import Buffer
from ops import LoadOps
from itertools import accumulate
from functools import reduce

class Tensor:
  def __init__(self, data, device=None, dtype=None, requires_grad=None):
    assert dtype is None or isinstance(dtype, Dtype), f"invalid dtype {dtype}"
    device = Device.canonicalize(device)

    # tensors have gradients, buffers do not
    self.grad: Optional[Tensor] = None
    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    self.requires_grad: Optional[bool] = requires_grad

    self._ctx: Optional[F.Function] = None
    if isinstance(data, Buffer): assert dtype is None or dtype == data.dtype, "dtype doesn't match"
    elif isinstance(data, (int, float)):
      data = Buffer(np.full(tuple(), data, dtype.np))
    elif data is None or data.__class__ is list:
      assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
      data = Buffer(np.array([] if data is None else data, (dtype.np or Tensor.default_type)))
    elif isinstance(data, np.ndarray):
      assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
      if data.shape == ():
        data = Buffer(np.full(tuple(), data, dtypes.from_np(data.dtype)))
      else:
        data = Buffer(data)
    self.data = data

  def __repr__(self):
    return f"<Tensor {self.data!r} on {self.device} with grad {(self.grad.data if self.grad else None)!r}"

  # *** properties ***
  @property
  def device(self) -> str: return self.data.device
  @property
  def shape(self) -> Tuple[int]: return self.data.shape
  @property
  def dtype(self) -> Dtype: return self.data.dtype
  @property
  def ndim(self) -> int: return len(self.shape)

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

  # *** creation helpers ****
  @staticmethod
  def full(shape:Tuple[int, ...], fill_value, **kwargs): return Tensor(fill_value, **kwargs).reshape([1]*len(new_shape := argfix(shape))).expand(new_shape)
  @staticmethod
  def zeros(*shape, **kwargs): return Tensor.full(argfix(*shape), 0, **kwargs)
  @staticmethod
  def ones(*shape, **kwargs): return Tensor.full(argfix(*shape), 1, **kwargs)
  @staticmethod
  def arange(start, stop=None, step=1, **kwargs):
    if stop == None: stop, start = start, 0
    return Tensor.full((math.ceil((stop-start)/step),), step, **kwargs).cumsum() + (start - step)
  @staticmethod
  def eye(dim:int, **kwargs): return Tensor.full((dim,1),1,**kwargs).pad(((0,0),(0,dim))).reshape(dim*(dim+1)).shrink(((0,dim*dim),)).reshape(dim,dim)

  def full_like(self, fill_value, **kwargs): return Tensor.full(self.shape, fill_value=fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
  def zeros_like(self, **kwargs): return self.full_like(0, **kwargs)
  def ones_like(self, **kwargs): return self.full_like(1, **kwargs)

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

  # op wrappers
  def __neg__(self) -> Tensor: return self.neg()
  def __add__(self, x) -> Tensor: return self.add(x)
  def __sub__(self, x) -> Tensor: return self.sub(x)
  def __mul__(self, x) -> Tensor: return self.mul(x)
  def __truediv__(self, x) -> Tensor: return self.div(x)
  def __matmul__(self, x) -> Tensor: return self.matmul(x)

  # *** ternary ***
  def where(self, x:Tensor, y:Tensor): return F.Where.apply(x, x, y)

  # *** reduce ***
  def _reduce(self, fxn:Type[F.Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False) -> Tensor:
    axis_: List[int] = list(range(len(self.shape))) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
    axis_ = [x if x>=0 else x+len(self.shape) for x in axis_]
    shape = tuple(s for i,s in enumerate(self.shape) if i not in axis_)
    if 0 in self.shape and 0 not in shape:
      v = {F.Sum: 0, F.Max: -float("inf")}[fxn]
      return Tensor.full(tuple(1 if s == 0 else s for s in self.shape) if keepdim else shape, v)
    ret = fxn.apply(self, new_shape=tuple([1 if i in axis_ else s for i,s in enumerate(self.shape)]))
    return ret if keepdim else ret.reshape(shape=shape)
  
  def sum(self, axis=None, keepdim=False): return self._reduce(F.Sum, axis, keepdim)
  def max(self, axis=None, keepdim=False): return self._reduce(F.Max, axis, keepdim)
  def min(self, axis=None, keepdim=False): return -((-self)._reduce(F.Max, axis, keepdim))
  def mean(self, axis=None, keepdim=False):
    out = self.sum(axis=axis, keepdim=keepdim)
    # return out.mul(prod(out.shape)/prod(self.shape)) if 0 not in self.shape else out
    return out.div(prod(self.shape)/prod(out.shape)) if 0 not in self.shape else out
  def std(self, axis=None, keepdim=False, correction=1):
    square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
    return square_sum.div(prod(self.shape)/prod(square_sum.shape)-correction).sqrt()
  # https://en.wikipedia.org/wiki/Softmax_function
  def _softmax(self, axis):
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)
  def softmax(self, axis=-1):
    _, e, ss = self._softmax(axis)
    return e.div(ss)
  def log_softmax(self, axis=-1):
    m, _, ss = self._softmax(axis)
    return m - ss.log()
  def argmax(self, axis=None, keepdim=False):
    if axis is None:
      idx = (self == self.max(axis)) * Tensor.arange(prod(self.shape)-1,-1,-1, dtype=dtypes.int32, device=self.device).reshape(self.shape)
      return prod(self.shape) - idx.max() - 1
    axis = axis+len(self.shape) if axis < 0 else axis
    m = (self == self.max(axis=axis, keepdim=True))
    idx = m * Tensor.arange(self.shape[axis]-1, -1, -1, dtype=dtypes.int32, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1
  def argmin(self, axis=None, keepdim=False): return (-self).argmax(axis=axis, keepdim=keepdim)


  # *** movement ***
  def _resolve_dim(self, dim:int, *, outer:bool=False) -> int:
    if not -max(1, self.ndim+outer) <= dim < max(1, self.ndim+outer):
      raise IndexError(f"{dim=} out of range {[-max(1, self.ndim+outer), max(1, self.ndim+outer)-1]}")
    return dim + self.ndim+outer if dim < 0 else dim

  def reshape(self, shape, *args) -> Tensor:
    new_shape = argfix(shape, *args)
    return F.Resahpe.apply(self, shape=tuple([-prod(self.shape)//prod(new_shape) if s == -1 else (s if s is not None else self.shape[i]) for i,s in enumerate(new_shape)]))
  def expand(self, shape, *args) -> Tensor: return F.Expand.apply(self, shape=tuple([x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))]))
  def permute(self, order, *args) -> Tensor: return F.Permute.apply(self, order=argfix(order, *args))
  def flip(self, axis, *args) -> Tensor: return F.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])
  
  def shrink(self, arg:Tuple[Optional[Tuple[int, int]], ...]) -> Tensor:
    F.Shrink.apply(self, arg=tuple(x if x is not None else (0,s) for x,s in zip(arg.self.shape))) if any(x is not None and x != (0,s) for x,s in zip(arg,self.shape)) else self
  
  def pad(self, arg:Tuple[Optional[Tuple[int, int]], ...], value:float=0.0) -> Tensor:
    if all(x is None or x == (0,0) for x in arg): return self
    ret = F.Pad.apply(self, arg=(narg:=tuple(x if x is not None else (0,0) for x in arg)))
    return ret if 0 == value else ret + Tensor.where(F.Pad.apply(Tensor.ones_like(self), arg=narg), 0, value)
  
  def gather(self:Tensor, idx:Tensor, dim:int) -> Tensor:
    assert idx.ndim == self.ndim, "self.ndim must equal idx.ndim"
    assert all(ix <= s if i != dim else True for i,(ix,s) in enumerate(zip(idx.shape, self.shape))), "all dim (except input dim) of idx.shape must be smaller than self.shape"
    if dim < 0: dim += self.ndim
    permarg = list(range(self.ndim))
    permarg = permarg[1:dim] + [permarg[0] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]]
    return ((idx == Tensor.arange(self.shape[dim], dtype=dtypes.int32, requires_grad=False, device=self.device)) * self.permute(*permarg).shrink(tuple([*[(0,s) for s in idx.shape[1:-1]], (0,self.shape[dim])])).unsqueeze(0)).sum(-1).transpose(0, dim)
  
  def cat(self, *args:Tensor, dim:int=0) -> Tensor:
    dim = (dim + len(self.shape)) if dim < 0 else dim
    assert all(len(y.shape) == len(self.shape) and all(y.shap[i] == s for i,s in enumerate(self.shape) if i != dim) for y in args)
    catargs = [self, *args]
    cat_dims = [s.shape[dim] for s in catargs]
    cat_dim_cumsum = [0, *accumulate(cat_dims)]
    slc = [[None for _ in self.shape] for _ in catargs]
    for d,k,s in zip(cat_dims, cat_dim_cumsum[:-1], slc):
      s[dim] = (k, cat_dim_cumsum[-1]-k-d)
    return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg,s in zip(catargs, slc)])
  
  def squeeze(self, dim:Optional[int]=None) -> Tensor:
    if dim is None: return self.reshape(tuple(dim for dim in self.shape if dim != 1))
    dim = self._resolve_dim(dim)
    return self if not self.ndim or self.shape[dim] != 1 else self.reshape(self.shape[:dim] + self.shape[dim+1])
  
  def unsqueeze(self, dim:int) -> Tensor:
    dim = self._resolve_dim(dim, outer=True)
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])



  

"""
*** high level tensor ops ***

# creation helpers
rand
randn
randint

# binary
pow

# reduce ops
avg_pool2d
max_pool2d
conv2d
cumsum
triu
tril

# movement ops
__getitem__
__setitem__
slice
stack
repeat
chunk

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