from __future__ import annotations
from typing import Optional, Final
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Dtype:
  itemsize: int
  name: str
  np: Optional[type]
  def __repr__(self): return f"dtypes.{self.name}"

class dtypes:
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: Dtype) -> bool: return x == dtypes.int32
  @staticmethod
  def is_float(x: Dtype) -> bool: return x == dtypes.float32
  @staticmethod
  def is_unsigned(x: Dtype) -> bool: return x == dtypes.uint8
  @staticmethod
  def from_np(x) -> Dtype: return DTYPES_DICT[np.dtype(x).name]
  bool: Final[Dtype] = Dtype(1, "bool", np.bool_)
  uint8: Final[Dtype] = Dtype(1, "unsigned char", np.uint8)
  int32: Final[Dtype] = Dtype(4, "int", np.int32)
  float32: Final[Dtype] = Dtype(4, "float", np.float32)

DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}