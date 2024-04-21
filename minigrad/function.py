import math
from buffer import Buffer
from ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps

class Function:
  def __init__(self):
    pass


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