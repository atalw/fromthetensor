# used teenygrad as the learning resource - https://github.com/tinygrad/teenygrad
from __future__ import annotations
import time, math
import numpy as np
from typing import Tuple, Optional, Union, Type, List, Sequence, Any, ClassVar
from lilgrad.dtype import Dtype, dtypes
from lilgrad.device import Device
from lilgrad.helpers import prod, argfix, make_pair, flatten, round_up
import lilgrad.function as F
from lilgrad.buffer import Buffer
from lilgrad.ops import LoadOps
from itertools import accumulate
from functools import reduce
from collections import defaultdict

class Tensor:
  training: ClassVar[bool] = False
  class train:
    def __init__(self, val=True): self.val = val
    def __enter__(self): self.prev, Tensor.training = Tensor.training, self.val
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any): Tensor.training = self.prev

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
      data = Buffer(np.full(tuple(), data, dtype.np if dtype else dtypes.float32.np))
    elif data is None or data.__class__ is list:
      assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
      data = Buffer(np.array([] if data is None else data, dtype.np if dtype else dtypes.float32.np))
    elif isinstance(data, bytes):
      data = Buffer(np.frombuffer(data, np.uint8))
    elif isinstance(data, np.ndarray):
      assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
      if data.shape == (): data = Buffer(np.full(tuple(), data, dtypes.from_np(data.dtype)))
      else: data = Buffer(data)
    else:  raise NotImplementedError()
    self.data = data

  def __repr__(self):
    return f"<Tensor {self.data!r} on {self.device} with grad {(self.grad.data if self.grad else None)!r}"
  
  def __hash__(self): return id(self)

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
  def cast(self, dtype:Dtype) -> Tensor: return F.Cast.apply(self, dtype=dtype) if self.dtype != dtype else self

  # *** convenience ***
  def numel(self) -> int: return prod(self.shape)
  def element_size(self) -> int: return self.dtype.itemsize
  def nbytes(self) -> int: return self.numel() * self.element_size()

  # *** creation helpers ****
  @staticmethod
  def _loadop(op, sz, device:Optional[str]=None, dtype:Optional[Dtype]=None, arg=None, **kwargs) -> Tensor:
    assert isinstance(sz, int), f"cannot create with symbolic size {sz}"
    return Tensor(Buffer.loadop(op, (sz,), dtype or dtypes.float32, Device.canonicalize(device), arg), dtype=dtype, device=device, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs) -> Tensor: return Tensor._loadop(LoadOps.EMPTY, prod((shape:=argfix(*shape))), **kwargs).reshape(shape)

  @staticmethod
  def full(shape:Tuple[int, ...], fill_value, **kwargs) -> Tensor: return Tensor(fill_value, **kwargs).reshape([1]*len(new_shape := argfix(shape))).expand(new_shape)

  @staticmethod
  def zeros(*shape, **kwargs) -> Tensor: return Tensor.full(argfix(*shape), 0, **kwargs)

  @staticmethod
  def ones(*shape, **kwargs) -> Tensor: return Tensor.full(argfix(*shape), 1, **kwargs)

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs) -> Tensor:
    if stop == None: stop, start = start, 0
    return Tensor.full((math.ceil((stop-start)/step),), step, **kwargs).cumsum() + (start - step)

  @staticmethod
  def eye(dim:int, **kwargs) -> Tensor: return Tensor.full((dim,1),1,**kwargs).pad(((0,0),(0,dim))).reshape(dim*(dim+1)).shrink(((0,dim*dim),)).reshape(dim,dim)

  _seed: int = int(time.time())
  @staticmethod
  def rand(*shape, **kwargs) -> Tensor:
    Tensor._seed += 1
    return Tensor._loadop(LoadOps.RAND, prod((shape:=argfix(*shape))), arg=Tensor._seed, **kwargs).reshape(shape)

  @staticmethod
  def randn(*shape, dtype:Optional[Dtype]=None, **kwargs) -> Tensor:
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    src = Tensor.rand((2, *argfix(*shape)), **{**kwargs, "dtype": dtypes.float32})
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt())

  @staticmethod
  def randint(*shape, low=0, high=10, **kwargs) -> Tensor: return Tensor.uniform(*shape, low=low, high=high, **kwargs).cast(dtypes.int32)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor: return (std * Tensor.randn(*shape, **kwargs)) + mean

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor: return ((high-low) * Tensor.rand(*shape, **kwargs)) + low

  @staticmethod
  def kaiming_uniform(*shape, a:float=0.01, **kwargs) -> Tensor:
    std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

  def full_like(self, fill_value, **kwargs) -> Tensor: return Tensor.full(self.shape, fill_value=fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
  def zeros_like(self, **kwargs) -> Tensor: return self.full_like(0, **kwargs)
  def ones_like(self, **kwargs) -> Tensor: return self.full_like(1, **kwargs)

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
  def sign(self): return self / (self.abs() + 1e-10)

  # activation functions
  def swish(self): return self * self.sigmoid()
  def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def sinh(self): return (self.exp() - self.neg().exp()) / 2
  def cosh(self): return (self.exp() + self.neg().exp()) / 2
  def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
  def quick_gelu(self): return self * (self * 1.702).sigmoid()
  def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()

  # *** binary ***
  def _broadcasted(self, y:Union[Tensor, float], reverse:bool=False) -> Tuple[Tensor, Tensor]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      if 0 in x.shape: return x, x.full_like(y)
      y = Tensor(y, device=self.device, requires_grad=False, dtype=self.dtype)
    if reverse: x, y = y, x
    if (xshape:=x.shape) == (yshape:=y.shape): return (x, y)

    shape_delta = len(xshape) - len(yshape)
    if shape_delta > 0: y = y.reshape((1,) * shape_delta + yshape)
    elif shape_delta < 0: x = x.reshape((1,) * -shape_delta + xshape)
    if (xshape:=x.shape) == (yshape:=y.shape): return (x, y)

    shape_ret = tuple([max(x, y) for x, y in zip(xshape, yshape)])
    if xshape != shape_ret: x = x.expand(shape_ret)
    if yshape != shape_ret: y = y.expand(shape_ret)
    return (x, y)

  def add(self, x:Union[Tensor, float], reverse=False) -> Tensor: return F.Add.apply(*self._broadcasted(x, reverse))
  def sub(self, x:Union[Tensor, float], reverse=False) -> Tensor: return F.Sub.apply(*self._broadcasted(x, reverse))
  def mul(self, x:Union[Tensor, float], reverse=False) -> Tensor: return F.Mul.apply(*self._broadcasted(x, reverse))
  def div(self, x:Union[Tensor, float], reverse=False) -> Tensor: return F.Div.apply(*self._broadcasted(x, reverse))
  def dot(self, w:Tensor) -> Tensor:
    n1, n2 = len(self.shape), len(w.shape)
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

  def __radd__(self, x) -> Tensor: return self.add(x, True)
  def __rsub__(self, x) -> Tensor: return self.sub(x, True)
  def __rmul__(self, x) -> Tensor: return self.mul(x, True)
  def __rtruediv__(self, x) -> Tensor: return self.div(x, True)
  def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)

  def __lt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, False))
  def __gt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, True))
  def __ge__(self, x) -> Tensor: return (self<x).logical_not()
  def __le__(self, x) -> Tensor: return (self>x).logical_not()
  def __eq__(self, x) -> Tensor: return F.Eq.apply(*self._broadcasted(x, True))
  def __ne__(self, x) -> Tensor: return (self==x).logical_not()

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
  
  def sum(self, axis=None, keepdim=False) -> Tensor: return self._reduce(F.Sum, axis, keepdim)
  def max(self, axis=None, keepdim=False) -> Tensor: return self._reduce(F.Max, axis, keepdim)
  def min(self, axis=None, keepdim=False) -> Tensor: return -((-self)._reduce(F.Max, axis, keepdim))
  def mean(self, axis=None, keepdim=False) -> Tensor:
    out = self.sum(axis=axis, keepdim=keepdim)
    # return out.mul(prod(out.shape)/prod(self.shape)) if 0 not in self.shape else out
    return out.div(prod(self.shape)/prod(out.shape)) if 0 not in self.shape else out
  def std(self, axis=None, keepdim=False, correction=1) -> Tensor:
    square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
    return square_sum.div(prod(self.shape)/prod(square_sum.shape)-correction).sqrt()
  # https://en.wikipedia.org/wiki/Softmax_function
  def _softmax(self, axis):
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)
  def softmax(self, axis=-1) -> Tensor:
    _, e, ss = self._softmax(axis)
    return e.div(ss)
  def log_softmax(self, axis=-1) -> Tensor:
    m, _, ss = self._softmax(axis)
    return m - ss.log()
  def argmax(self, axis=None, keepdim=False) -> Tensor:
    if axis is None:
      idx = (self == self.max(axis)) * Tensor.arange(prod(self.shape)-1,-1,-1, dtype=dtypes.int32, device=self.device).reshape(self.shape)
      return prod(self.shape) - idx.max() - 1
    axis = axis+len(self.shape) if axis < 0 else axis
    m = (self == self.max(axis=axis, keepdim=True))
    idx = m * Tensor.arange(self.shape[axis]-1, -1, -1, dtype=dtypes.int32, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1
  def argmin(self, axis=None, keepdim=False) -> Tensor: return (-self).argmax(axis=axis, keepdim=keepdim)

  def _cumsum(self, axis:int=0, _first_zero=False) -> Tensor: return self.transpose(axis,-1).pad2d((self.shape[axis]-int(not _first_zero),0))._pool((self.shape[axis],)).sum(-1).transpose(axis,-1)
  def cumsum(self, axis:int=0) -> Tensor:
    # TODO: someday the optimizer will find this on it's own
    # for now this is a two stage cumsum
    SPLIT = 256
    if self.shape[axis] <= SPLIT*2: return self._cumsum(axis)
    ret = self.transpose(axis,-1).pad2d((round_up(self.shape[axis], SPLIT)-self.shape[axis], 0))
    ret = ret.reshape(*ret.shape[0:-1], ret.shape[-1]//SPLIT, SPLIT)._cumsum(-1)
    base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
    base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])
    def fix(x:Tensor): return x.reshape(*ret.shape[0:-2], ret.shape[-2] * ret.shape[-1])[..., -self.shape[axis]:].transpose(axis,-1)
    return fix(ret) + fix(base_add)

  # *** movement ***
  def _resolve_dim(self, dim:int, *, outer:bool=False) -> int:
    if not -max(1, self.ndim+outer) <= dim < max(1, self.ndim+outer):
      raise IndexError(f"{dim=} out of range {[-max(1, self.ndim+outer), max(1, self.ndim+outer)-1]}")
    return dim + self.ndim+outer if dim < 0 else dim


    # - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  # - A slice i:j returns the elements with indices in [i, j)
  #    - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
  #    - Negative values for i and j are taken relative to the end of the sequence
  #    - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
  # - Indexing with None on a given axis will add a new dimension of size one before that axis
  # - Empty slices are not allowed (tensors with 0s in shape have to be supported first, for all backends).
  # - For a slice [i:j:k] finding the correct indices is delegated to slice.indices(len).
  # - Strides > 1 and < 0 are now allowed!:
  #    - This works by applying Shrink -> [[Flip -> ] Pad -> Reshape -> Shrink] -> Reshape (ops in brackets are optional)
  #    - Idea of stride < 0 support:
  #        - Do the slice first, flip the axes were slice.step is negative, do slice.step -> -slice.step. Go to steps below.
  #    - Idea of stride `s` > 1 support (Pad -> Reshape -> Shrink):
  #        - Instead of doing [::s] on axis [dim_sz], do [:, 0] on axes [dim_sz_padded // s, s].
  #        - So pad dim_sz with as many zeros as needed (dim_sz -> dim_sz_padded) so that reshape to [dim_sz_padded // s, s]
  #          is possible.
  #        - Apply Shrink to do the slice [:, 0] on axes of shapes [dim_sz_padded // s, s].
  # - Fancy indexing and combined indexing is supported
  #    - Combined indexing works by letting regular slicing finish first -> computing the resulting dims w.r.t to Tensors passed in -> fancy indexing
  #    - Any Tensors passed in __getitem__ will perform (CMPEQ with arange -> MUL with self -> SUM_REDUCE) iteratively
  #        - The first iteration will expand the dim of self while consecutive iterations will reduce the dim
  #    - There's a special case where a permute is needed at the end:
  #        - if first Tensor passed in (expand dims) is not at dim 0
  #        - and following Tensors does not follow consecutively to the end of fancy indexing's dims
  def __getitem__(self, val) -> Tensor: # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
    def normalize_int(e, i, dim_sz):
      if -dim_sz <= e < dim_sz: return e if e != -1 else dim_sz-1
      raise IndexError(f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}")

    orig_slices = list(val) if isinstance(val, tuple) else [val]
    count = defaultdict(list)
    for i,v in enumerate(orig_slices): count[type(v)].append(i)

    if (num_slices := len(count[int]) + len(count[slice]) + len(count[Tensor])) > len(self.shape): raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")
    if len(ellipsis_found := count[type(Ellipsis)]) > 1: raise IndexError("an index can only have a single ellipsis ('...')")

    ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
    orig_slices[ellipsis_idx:ellipsis_idx+1] = [slice(None)] * (len(self.shape) - num_slices)

    valid_slices = [v for v in orig_slices if v is not None]
    valid_slices = [v if isinstance(v, slice) else slice(y_ := normalize_int(v, i, dim_sz), y_+1) if isinstance(v, int) else slice(None) for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))]

    start, stop, strides = zip(*y) if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]) else ((), (), ())
    new_slice = tuple(((0, 0) if e < s else (s, e)) if st > 0 else ((0, 0) if e > s else (e+1, s+1)) for s, e, st in zip(start, stop, strides))
    sliced_tensor = self.shrink(new_slice).flip(axis=[i for i, s in enumerate(strides) if s < 0])
    new_shape = sliced_tensor.shape
    if any(abs(s) != 1 for s in strides):
      strides = tuple(abs(s) for s in strides)
      # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
      padded_tensor = sliced_tensor.pad(tuple((0, s-(dim_sz % s) if dim_sz % s != 0 else 0) for s, dim_sz in zip(strides, sliced_tensor.shape)))
      # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
      reshaped_tensor = padded_tensor.reshape(flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides)))
      new_shape = reshaped_tensor.shape[::2]
      # Shrink: do [:, 0]
      sliced_tensor = reshaped_tensor.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in new_shape)))

    final_shape, it_shape, dim, tensors, dim_collapsed = [], iter(new_shape), [], [], 0
    for i,s in enumerate(orig_slices):
      if s is None: final_shape.append(1)
      else: # s is int or slice or Tensor
        dim_shape = next(it_shape)
        if isinstance(s, int):
          dim_collapsed += 1
        else:
          assert isinstance(dim_shape, int), f"does not support symbolic shape {dim_shape}"
          final_shape.append(dim_shape)
          if isinstance(s, Tensor):
            tensors.append(s)
            dim.append(i-dim_collapsed)
    ret = sliced_tensor.reshape(tuple(final_shape))

    if tensors: # Fancy/tensor indexing
      # normalize idx
      # TODO: first contiguous fixes torch+cpu_only CI, but it causes llvm to fail. Second one fixes llvm
      idx = [t.sign().__neg__().relu() * ret.shape[d] + t for d,t in zip(dim, tensors)]
      max_dim = max(i.ndim for i in idx)
      # compute sum_dim, arange, and idx
      sum_dim = [d if n==0 else d+max_dim-n for n,d in enumerate(dim)]
      arange = [Tensor.arange(ret.shape[d], dtype=dtypes.int32, requires_grad=False, device=self.device).reshape(*[1]*sd, ret.shape[d], *[1]*(ret.ndim + max_dim - n - sd - 1)) for n,(sd,d) in enumerate(zip(sum_dim, dim))]
      first_idx = [idx[0].reshape(*[1]*dim[0], *[1]*(1 + max_dim - idx[0].ndim), *idx[0].shape, *[1]*(ret.ndim - dim[0] - 1))]
      rest_idx = [i.reshape(*[1]*dim[0], *[1]*(max_dim - i.ndim), *i.shape, *[1]*(ret.ndim - dim[0] - n)) for n,i in enumerate(idx[1:], 1)]
      idx = first_idx + rest_idx
      ret = ret.reshape(*ret.shape[:sum_dim[0]+1], *[1]*max_dim, *ret.shape[sum_dim[0]+1:])
      # iteratively fancy index
      for a,i,sd in zip(arange, idx, sum_dim): ret = (a==i).mul(ret).sum(sd)
      # special permute case
      if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1]+1)):
        ret_dims = list(range(ret.ndim))
        ret = ret.permute(ret_dims[dim[0]:dim[0]+max_dim] + ret_dims[:dim[0]] + ret_dims[dim[0]+max_dim:])
    return ret

  def __setitem__(self,s,v): return self.__getitem__(s).assign(v)

  def reshape(self, shape, *args) -> Tensor:
    new_shape = argfix(shape, *args)
    return F.Reshape.apply(self, shape=tuple([-prod(self.shape)//prod(new_shape) if s == -1 else (s if s is not None else self.shape[i]) for i,s in enumerate(new_shape)]))

  def expand(self, shape, *args) -> Tensor: return F.Expand.apply(self, shape=tuple([x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))]))

  def permute(self, order, *args) -> Tensor: return F.Permute.apply(self, order=argfix(order, *args))

  def flip(self, axis, *args) -> Tensor: return F.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])
  
  def shrink(self, arg:Tuple[Optional[Tuple[int, int]], ...]) -> Tensor:
    return F.Shrink.apply(self, arg=tuple(x if x is not None else (0,s) for x,s in zip(arg, self.shape))) if any(x is not None and x != (0,s) for x,s in zip(arg,self.shape)) else self
  
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
  
  def repeat(self, repeats:Sequence[int]) -> Tensor:
    base_shape = (1,) * (len(repeats) - self.ndim) + self.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r*s for r,s in zip(repeats, base_shape)]
    return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)

  # (padding_left, padding_right, padding_top, padding_bottom)
  def pad2d(self, padding:Sequence[int], value:float=0) -> Tensor:
    slc = [(-p0, s+p1) for p0,p1,s in zip(padding[::2], padding[1::2], self.shape[::-1])][::-1]
    return self.slice([(0,s) for s in self.shape[:-(len(padding)//2)]] + slc, value=value)

  def transpose(self, ax1=1, ax2=0) -> Tensor:
    order = list(range(self.ndim))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return self.permute(order)
  
  def flatten(self, start_dim=0, end_dim=-1):
    start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim+1]), ) + self.shape[end_dim+1:])

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
  
  def sparse_categorical_crossentropy(self, Y:Tensor, ignore_index=-1, label_smoothing=0.0) -> Tensor:
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    # NOTE: self is a logits input
    log_probs, loss_mask = self.log_softmax(), (Y != ignore_index)
    y_counter = Tensor.arange(self.shape[-1], requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    smoothing = -1 * label_smoothing * (log_probs.mean(-1) * loss_mask).sum() / loss_mask.sum()
    return (1 - label_smoothing) * (log_probs * y).sum() / loss_mask.sum() + smoothing

  def _pool(self, k_:Tuple[int, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    noop_, i_ = [None] * len(self.shape[:-len(k_)]), self.shape[-len(k_):]
    if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
      o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
      # repeats such that we don't need padding
      xup = self.repeat([1]*len(noop_) + [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)])
      # slice by dilation
      xup = xup.shrink(tuple(noop_ + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])).reshape(noop_ + flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
      # handle stride
      xup = xup.shrink(noop_ + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_))).reshape(noop_ + flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
      xup = xup.shrink(noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_))).reshape(noop_ + flatten((k,o) for k,o in zip(k_, o_)))
      # permute to move reduce to the end
      return xup.permute(*range(len(noop_)), *[len(noop_)+i*2+1 for i in range(len(i_))], *[len(noop_)+i*2 for i in range(len(i_))])
    # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
    o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
    xup = self.pad(tuple(noop_ + [(0, max(0,o*s-i)) for i,o,s in zip(i_, o_, s_)])).shrink(tuple(noop_ + [(0,o*s) for o,s in zip(o_, s_)]))
    xup = xup.reshape(noop_ + flatten(((o,s) for o,s in zip(o_, s_))))
    xup = xup.shrink(noop_ + flatten(((0,o), (0,k)) for o,k in zip(o_, k_)))
    return xup.permute(*range(len(noop_)), *[len(noop_)+i*2 for i in range(len(i_))], *[len(noop_)+i*2+1 for i in range(len(i_))])

  def avg_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).mean(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))
  def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, stride=1, dilation=1, padding=0) -> Tensor:
    groups = 1
    (bs,_cin), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert len(self.shape) == len(weight.shape), f"input tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({cin} vs. {_cin})"
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"
    _padding = [padding]*2*len(HW) if isinstance(padding, int) else (padding if len(padding) == 2*len(HW) else [p for p in padding for _ in range(2)][::-1])
  
    # conv2d is a pooling op (with padding)
    x = self.pad2d(_padding)._pool(HW, stride, dilation) # (bs, cin, oy, ox, H, W) [oy = output y]
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1:
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

# binary
pow

# reduce ops
cumsum
triu
tril

# movement ops
__getitem__
__setitem__
stack
repeat
chunk

# functional
scaled_dot_product_attention
"""