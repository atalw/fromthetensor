from __future__ import annotations
import math
from typing import Tuple, Optional, Type
from lilgrad.buffer import Buffer
import lilgrad.tensor as tensor 
from lilgrad.dtype import Dtype
from lilgrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps
from lilgrad.helpers import argsort

"""
forward and backward passes of low level ops

"""

class Function:
  def __init__(self, device:str, *tensors:tensor.Tensor):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:tensor.Tensor, **kwargs) -> tensor.Tensor:
    ctx = fxn(x[0].device, *x)
    ret = tensor.Tensor(ctx.forward(*[t.data for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
    if ctx.requires_grad and not tensor.Tensor.no_grad: ret._ctx = ctx
    return ret


class Cast(Function):
  def forward(self, x:Buffer, dtype:Dtype, bitcast:bool=False) -> Buffer:
    self.input_dtype, self.bitcast = x.dtype, bitcast
    return x.cast(dtype, bitcast)

  def backward(self, grad_output:Buffer) -> Buffer:
    return grad_output.cast(self.input_dtype, self.bitcast)

# *** unary ops ***

class Zero(Function):
  def forward(self, x:Buffer) -> Buffer: return x.const(0)
  def backward(self, grad:Buffer) -> Buffer: return grad.const(0)

class Exp(Function):
  def forward(self, x:Buffer) -> Buffer:
    # e^x = 2^(x * log2(e))
    # log2(e) = loge(e)/loge(2)
    self.ret = x.e(BinaryOps.MUL, x.const(1/math.log(2))).e(UnaryOps.EXP2)
    return self.ret

  def backward(self, grad:Buffer) -> Buffer:
    # d(e^x)/x = e^x
    return self.ret.e(BinaryOps.MUL, grad)

class Log(Function):
  def forward(self, x:Buffer) -> Buffer:
    self.x = x
    # loga(x) = logb(x) / logb(a)
    # if a = 2, b = e
    # logb(x) = log2(x) * loge(2)
    return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))

  def backward(self, grad:Buffer) -> Buffer:
    # dlog(fx)/dx = (d(fx)/dx) / fx
    return grad.e(BinaryOps.DIV, self.x)

class Sin(Function):
  def forward(self, x:Buffer) -> Buffer:
    self.x = x
    return x.e(UnaryOps.SIN)

  def backward(self, grad:Buffer) -> Buffer:
    # cos(x) = sin(90-x)
    return self.x.const(math.pi/2).e(BinaryOps.SUB, self.x).e(UnaryOps.SIN).e(BinaryOps.MUL, grad)

class Sqrt(Function):
  def forward(self, x:Buffer) -> Buffer:
    self.ret = x.e(UnaryOps.SQRT)
    return self.ret

  def backward(self, grad:Buffer) -> Buffer:
    # d(x^1/2)/x = 1/(2*x^(1/2))
    return grad.e(BinaryOps.DIV, self.ret.const(2).e(BinaryOps.MUL, self.ret))

class Neg(Function):
  def forward(self, x:Buffer) -> Buffer: return x.e(UnaryOps.NEG)
  def backward(self, grad:Buffer) -> Buffer: return grad.e(UnaryOps.NEG)

class Relu(Function):
  def forward(self, x:Buffer) -> Buffer:
    self.ret = x.const(0).e(BinaryOps.MAX, x)
    return self.ret

  def backward(self, grad:Buffer) -> Buffer:
    return self.ret.const(0).e(BinaryOps.CMPLT, self.ret).e(BinaryOps.MUL, grad)

class Sigmoid(Function):
  def forward(self, x:Buffer) -> Buffer:
    # f(x) = 1/(1 + e^(-x))
    self.ret = x.const(1).e(BinaryOps.DIV, x.const(1).e(BinaryOps.ADD, x.e(BinaryOps.MUL, x.const(-1/math.log(2))).e(UnaryOps.EXP2)))
    return self.ret

  def backward(self, grad:Buffer) -> Buffer:
    # https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    return self.ret.e(BinaryOps.MUL, self.ret.cost(1).e(BinaryOps.SUB, self.ret)).e(BinaryOps.MUL, grad)

# *** binary ops ***

class Less(Function):
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    return x.e(BinaryOps.CMPLT, y)

class Eq(Function):
  def forward(self, x:Buffer, y:Buffer) -> Buffer: return x.e(BinaryOps.CMPEQ, y)
  def backward(self, grad:Buffer) -> Tuple[Optional[Buffer], Optional[Buffer]]: return None, None

class Add(Function):
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    return x.e(BinaryOps.ADD, y)

  def backward(self, grad:Buffer) -> Tuple[Optional[Buffer], Optional[Buffer]]:
    return grad if self.needs_input_grad[0] else None, grad if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    return x.e(BinaryOps.SUB, y)

  def backward(self, grad:Buffer) -> Tuple[Optional[Buffer], Optional[Buffer]]:
    return grad if self.needs_input_grad[0] else None, grad.e(UnaryOps.NEG) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    self.x, self.y = x, y
    return x.e(BinaryOps.MUL, y)

  def backward(self, grad:Buffer) -> Tuple[Optional[Buffer], Optional[Buffer]]:
    return grad.e(BinaryOps.MUL, self.y) if self.needs_input_grad[0] else None, \
           grad.e(BinaryOps.MUL, self.x) if self.needs_input_grad[1] else None

class Div(Function):
  def forward(self, x:Buffer, y:Buffer) -> Buffer:
    self.x, self.y = x, y
    return x.e(BinaryOps.DIV, y)

  def backward(self, grad:Buffer) -> Tuple[Optional[Buffer], Optional[Buffer]]:
    # d(x/y)/dy = x/(y^2)
    return grad.e(BinaryOps.DIV, self.y) if self.needs_input_grad[0] else None, \
           grad.e(UnaryOps.NEG).e(BinaryOps.MUL, self.x).e(BinaryOps.DIV, self.y.e(BinaryOps.MUL, self.y)) if self.needs_input_grad[1] else None

# *** ternary ops ***

class Where(Function):
  def forward(self, x:Buffer, y:Buffer, z:Buffer) -> Buffer:
    self.x = x
    return x.e(TernaryOps.WHERE, y, z)

  def backward(self, grad:Buffer) -> Tuple[None, Optional[Buffer], Optional[Buffer]]:
    return None, \
           self.x.e(TernaryOps.WHERE, grad, grad.const(0)) if self.needs_input_grad[1] else None, \
           self.x.e(TernaryOps.WHERE, grad.const(0, grad)) if self.needs_input_grad[2] else None

# *** reduce ops ***

class Sum(Function):
  def forward(self, x:Buffer, new_shape:Tuple[int, ...]) -> Buffer:
    self.input_shape = x.shape
    return x.r(ReduceOps.SUM, new_shape)

  def backward(self, grad:Buffer) -> Buffer:
    return grad.m(MovementOps.EXPAND, self.input_shape)

class Max(Function):
  def forward(self, x:Buffer, new_shape:Tuple[int, ...]) -> Buffer:
    self.x, self.ret = x, x.r(ReduceOps.MAX, new_shape)
    return self.ret

  def backward(self, grad:Buffer) -> Buffer:
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = self.x.const(1.0).e(BinaryOps.SUB, self.x.e(BinaryOps.CMPLT, self.ret.m(MovementOps.EXPAND, self.x.shape)))
    div = max_is_1s.r(ReduceOps.SUM, grad.shape).m(MovementOps.EXPAND, self.x.shape)
    return max_is_1s.e(BinaryOps.DIV, div).e(BinaryOps.MUL, grad.m(MovementOps.EXPAND, self.x.shape))

# *** movement ops ***

class Expand(Function):
  def forward(self, x:Buffer, shape:Tuple[int, ...]) -> Buffer:
    self.input_shape = x.shape
    return x.m(MovementOps.EXPAND, shape)

  def backward(self, grad:Buffer) -> Buffer:
    return grad.r(ReduceOps.SUM, self.input_shape)

class Reshape(Function):
  def forward(self, x:Buffer, shape:Tuple[int, ...]) -> Buffer:
    self.input_shape = x.shape
    return x.m(MovementOps.RESHAPE, shape)

  def backward(self, grad:Buffer) -> Buffer:
    return grad.m(MovementOps.RESHAPE, self.input_shape)

class Permute(Function):
  def forward(self, x:Buffer, order:Tuple[int, ...]) -> Buffer:
    self.input_order = order
    return x.m(MovementOps.PERMUTE, order)

  def backward(self, grad:Buffer) -> Buffer:
    return grad.m(MovementOps.PERMUTE, argsort(self.input_order))

class Pad(Function):
  def forward(self, x:Buffer, arg:Tuple[Tuple[int, int], ...]) -> Buffer:
    self.narg = tuple([(p[0], s+p[0]) for s,p in zip(x.shape, arg)])
    return x.m(MovementOps.PAD, arg)

  def backward(self, grad:Buffer) -> Buffer:
    return grad.m(MovementOps.SHRINK, self.narg)

class Shrink(Function):
  def forward(self, x:Buffer, arg:Tuple[Tuple[int, int], ...]) -> Buffer:
    self.narg = tuple([(p[0], s-p[1]) for s,p in zip(x.shape, arg)])
    return x.m(MovementOps.SHRINK, arg)

  def backward(self, grad:Buffer) -> Buffer:
    return grad.m(MovementOps.PAD, self.narg)

class Flip(Function):
  def forward(self, x:Buffer, axis:Tuple[int, ...]) -> Buffer:
    self.arg = tuple([-1 if i in set(axis) else 1 for i in range(len(x.shape))])
    return x.m(MovementOps.STRIDE, self.arg)

  def backward(self, grad:Buffer) -> Buffer:
    return grad.m(MovementOps.STRIDE, self.arg)
