from __future__ import annotations
import numpy as np
from lilgrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps, LoadOps
from lilgrad.dtype import dtypes

class Buffer:
  device = "CPU"

  def __init__(self, buf: np.ndarray): self._np = buf

  @property
  def dtype(self) : return dtypes.from_np(self._np.dtype)

  @property
  def shape(self): return self._np.shape
  def __repr__(self): return f"<B {self.shape} {self.dtype}"

  def const(self, x) -> Buffer: return Buffer(np.full_like(self._np, x))

  # @staticmethod
  # def loadop(op, shape, dtype, device, arg=None, src=None) -> Buffer:
  #   if op == LoadOps.RAND: return Buffer(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
  #   elif op == LoadOps.CONST: return Buffer(np.full(shape, arg, dtype=dtype.np))
  #   elif op == LoadOps.EMPTY: return Buffer(np.empty(shape, dtype=dtype.np))
  #   else: raise NotImplementedError(op)

  # element-wise ops
  def e(self, op, *srcs:Buffer):
    if op == UnaryOps.NEG: ret = -self._np
    elif op == UnaryOps.EXP2: ret = np.exp2(self._np)
    elif op == UnaryOps.LOG2: ret = np.log2(self._np)
    elif op == UnaryOps.SIN: ret = np.sin(self._np)
    elif op == UnaryOps.SQRT: ret = np.sqrt(self._np)
    elif op == BinaryOps.ADD: ret = self._np + srcs[0]._np
    elif op == BinaryOps.SUB: ret = self._np - srcs[0]._np
    elif op == BinaryOps.MUL: ret = self._np * srcs[0]._np
    elif op == BinaryOps.DIV: ret = self._np / srcs[0]._np
    elif op == BinaryOps.MAX: ret = np.maximum(self._np, srcs[0]._np)
    elif op == TernaryOps.WHERE: ret = np.where(self._np, srcs[0]._np, srcs[1]._np)
    else: raise NotImplementedError(op)
    return Buffer(ret.astype(self.dtype.np, copy=False))
  
  # reduce ops
  def r(self, op, new_shape):
    assert len(self.shape) == len(new_shape, "reduce shapes must have same dimesions")
    axis = tuple(i for i,(a,b) in enumerate(zip(self.shape, new_shape)) if a != b)
    if op == ReduceOps.SUM: return Buffer(self._np.sum(axis, dtype=self._np.dtype, keepdims=True))
    elif op == ReduceOps.MAX: return Buffer(self._np.max(axis, keepdims=True))
    else: raise NotImplementedError(op)

  # movement ops
  def m(self, op, arg):
    if op == MovementOps.RESHAPE: return Buffer(self._np.reshape(arg))
    elif op == MovementOps.EXPAND: return Buffer(np.broadcast_to(self._np, arg))
    elif op == MovementOps.SHRINK: return Buffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)])
    elif op == MovementOps.PERMUTE: return Buffer(self._np.transpose(arg))
    elif op == MovementOps.PAD: return Buffer(np.pad(self._np, arg))
    elif op == MovementOps.STRIDE: return Buffer(self._np[tuple(slice(None, None, i) for i in arg)])