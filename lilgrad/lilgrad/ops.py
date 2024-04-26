from enum import Enum, auto

"""
basic low-level ops
all high level tensor ops will boil down to these
"""

class UnaryOps(Enum): 
  EXP2 = auto()
  LOG2 = auto()
  SIN = auto()
  SQRT = auto()
  RECIP = auto()
  NEG = auto()

class BinaryOps(Enum):
  ADD = auto()
  SUB = auto()
  MUL = auto()
  DIV = auto()
  MAX = auto()
  MOD = auto()
  CMPLT = auto()
  CMPEQ = auto()

class TernaryOps(Enum):
  WHERE = auto()

class ReduceOps(Enum):
  SUM = auto()
  MAX = auto()

class MovementOps(Enum):
  RESHAPE = auto()
  PERMUTE = auto()
  EXPAND = auto()
  PAD = auto()
  SHRINK = auto()
  STRIDE = auto()

class LoadOps(Enum):
  RAND = auto()
  CONST = auto()
  EMPTY = auto()