# used teenygrad as the learning resource - https://github.com/tinygrad/teenygrad
from __future__ import annotations
import time, math
import numpy as np
from typing import Tuple, Optional, Union, Type, List, Sequence
from dtype import Dtype, dtypes
from device import Device
from helpers import prod, argfix, make_pair, flatten
import function as F
from buffer import Buffer
from ops import LoadOps
from itertools import accumulate, reduce

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
    _seed: int = int(time.time())

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
  @property
  def T(self) -> Tensor: return self.transpose()

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
  def assign(self, x) -> Tensor:
    self.data = x.data
    return self
  def detach(self) -> Tensor: return Tensor(self.data, device=self.device, requires_grad=False)
  def numpy(self) -> np.ndarray: return self.detach().data

  # *** convenience ***
  def numel(self) -> int: return prod(self.shape)
  def element_size(self) -> int: return self.dtype.itemsize
  def nbytes(self) -> int: return self.numel() * self.element_size()

  # *** creation helpers ****
  @staticmethod
  def _loadop(op, sz, device:Optional[str]=None, dtype:Optional[Dtype]=None, arg=None, **kwargs):
    assert isinstance(sz, int), f"cannot create with symbolic size {sz}"
    return Tensor(Buffer.loadop(op, (sz,), Tensor.default_type if dtype is None else dtype, Device.canonicalize(device), arg), dtype=dtype, device=device, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs): return Tensor._loadop(LoadOps.EMPTY, prod((shape:=argfix(*shape))), **kwargs).reshape(shape)

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

  @staticmethod
  def rand(*shape, **kwargs):
    Tensor._seed += 1
    return Tensor._loadop(LoadOps.RAND, prod((shape:=argfix(*shape))), arg=Tensor._seed, **kwargs).reshape(shape)  @staticmethod

  @staticmethod
  def randn(*shape, dtype:Optional[Dtype]=None, **kwargs):
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    src = Tensor.rand((2, *argfix(*shape)), **{**kwargs, "dtype": dtypes.float32})
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or dtypes.default_float)

  @staticmethod
  def randint(*shape, low=0, high=10, **kwargs): return Tensor.uniform(*shape, low=low, high=high, dtype=dtypes.int32, **kwargs)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, **kwargs): return (std * Tensor.randn(*shape, **kwargs)) + mean

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, **kwargs): return ((high-low) * Tensor.rand(*shape, **kwargs)) + low

  @staticmethod
  def kaiming_uniform(*shape, a:float=0.01, **kwargs):
    std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

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
  def maximum(self, x:Union[Tensor, float]) -> Tensor:
    return Tensor.where(self < x, x, (self == x).where((self*0.5 + x*0.5), self))
  def minimum(self, x:Union[Tensor, float]) -> Tensor: return -((-self).maximum(-x))

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

  def slice(self, arg:Sequence[Optional[Tuple[int, int]]], value:float=0) -> Tensor:
    arg_ = tuple(a if a is not None else (0, s) for s,a in zip(self.shape, arg))
    padding = tuple((max(0, -l), max(0, r-s)) for s,(l,r) in zip(self.shape, arg_))
    return self.pad(padding, value=value).shrink(tuple((l + pl, r + pl) for (l,r),(pl,_) in zip(arg_, padding)))

  # (padding_left, padding_right, padding_top, padding_bottom)
  def pad2d(self, padding:Sequence[int], value:float=0) -> Tensor:
    slc = [(-p0, s+p1) for p0,p1,s in zip(padding[::2], padding[1::2], self.shape[::-1])][::-1]
    return self.slice([(0,s) for s in self.shape[:-(len(padding)//2)]] + slc, value=value)

  def transpose(self, ax1=1, ax2=0) -> Tensor:
    order = list(range(self.ndim))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return self.permute(order)

  # *** functional ****

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  # https://arxiv.org/pdf/1607.06450.pdf
  def layernorm(self, axis=-1, eps:float=1e-5) -> Tensor:
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rqsrt())
    # where is g? what is eps?

  # https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
  def dropout(self, p:float=0.5) -> Tensor:
    if not Tensor.training or p == 0: return self
    # scaling - https://cs231n.github.io/neural-networks-2/
    return self * (Tensor.rand(*self.shape, requires_grad=False, device=self.device) >= p) * (1/(1.0 - p))

  def one_hot(self, num_classes:int) -> Tensor:
    return Tensor.where(self[..., None] == Tensor.arange(num_classes, requires_grad=False, device=self.device), 1, 0)

  def binary_crossentropy(self, y:Tensor) -> Tensor:
    return (-y*self.log() + (1-y)*(1-self).log()).mean()

  def binary_crossentropy_logits(self, y:Tensor) -> Tensor:
    return (self.maximum(0) - (y*self) + (1+self.abs().neg().exp()).log()).mean()

  def _pool(self, kernel:Tuple[int, ...], stride:Union[Tuple[int, ...], int]=1):
    assert len(self.shape) >= len(kernel), f"can't pool {self.shape} with {kernel}"
    _s, _k = make_pair(stride, len(kernel)), kernel
    assert len(_k) == len(_s), f"stride mismatch kernel:{_k} stride:{_s}"
    if any(k > s for k,s in zip(_k, _s)): raise NotImplementedError()
    _noop, _i = [None] * len(self.shape[:-len(_k)]), self.shape[-len(_k):]
    _o = [(i+(s-k))//s for i,s,k in zip(_i, _s, _k)]
    xup = self.pad(tuple(_noop + [(0, max(0,o*s-i)) for i,o,s in zip(_i, _o, _s)])).shrink(tuple(_noop + [(0,o*s) for o,s in zip(_o, _s)]))
    xup = xup.reshape(_noop + flatten(((o,s) for o,s in zip(_o, _s))))
    xup = xup.shrink(_noop + flatten(((0,o), (0,k)) for o,k in zip(_o, _k)))
    return xup.permute(*range(len(_noop)), *[len(_noop)+i*2 for i in range(len(_i))], *[len(_noop)+i*2+1 for i in range(len(_i))])

  def avg_pool2d(self, kernel_size=(2,2), stride=None): return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size).mean(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))
  def max_pool2d(self, kernel_size=(2,2), stride=None): return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, stride=1, padding=0) -> Tensor:
    groups = 1
    (bs,_cin), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert len(self.shape) == len(weight.shape), f"input tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({cin} vs. {_cin})"
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"
    _padding = [padding]*2*len(HW) if isinstance(padding, int) else (padding if len(padding) == 2*len(HW) else [p for p in padding for _ in range(2)][::-1])
  
    # conv2d is a pooling op (with padding)
    x = self.pad2d(_padding)._pool(HW, stride) # (bs, cin, oy, ox, H, W) [oy = output y]
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1:
      # normal conv
      x = x.reshape(bs, groups, rcout, *[1] * len(oyx), cin, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])

      # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True).reshape(bs, cout, *oyx)
      return ret if bias is None else ret.add(bias.resahpe(1, -1, *[1] * len(HW)))

    raise NotImplementedError()
  
  def batchnorm(self, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor, axis:Union[int,Tuple[int,...]]=1) -> Tensor:
    axis_ = argfix(axis)
    shape = tuple(s if ax in axis else 1 for ax,s in enumerate(self.shape))
    x = self - mean.reshape(shape)
    if weight is not None: x = x * weight.reshape(shape)
    ret = x.mul(invstd.reshape(shape) if len(invstd.shape) == len(axis_) else invstd)
    return ret if bias is None else (ret + bias.reshape(shape))


"""
*** high level tensor ops ***

# creation helpers
rand
randn
randint

# binary
pow

# reduce ops
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
batchnorm
dropout
scaled_dot_product_attention
sparse_categorial_crossentropy
"""