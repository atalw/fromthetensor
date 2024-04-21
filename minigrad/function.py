import math
from typing import Tuple, Optional
from buffer import Buffer
from tensor import Tensor
from ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps

class Function:
  def __init__(self, device:str, *tensors:Tensor):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {type(self)}")

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


# *** binary ops ***

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

  def backward(self, grad:Buffer) -> Tuple[Optional[Buffer], Optional[Buffer]]:
    return None, \
           self.x.e(TernaryOps.WHERE, grad, grad.const(0)) if self.needs_input_grad[1] else None, \
           self.x.e(TernaryOps.WHERE, grad.const(0, grad)) if self.needs_input_grad[2] else None


"""
forward and backward passes of low level ops

# unary ops
zero
neg
sin
relu
log
exp
sqrt
sigmoid

# binary ops
less
add
sub
mul
div

# ternary ops
where

# reduce ops
sum
max

# movement ops
expand
reshape
permute
pad
shrink
flip


"""