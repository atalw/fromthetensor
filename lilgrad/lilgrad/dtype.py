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
  uint8: Final[Dtype] = Dtype(1, "unsigned char", np.int8)
  int32: Final[Dtype] = Dtype(4, "int", np.int32)
  float32: Final[Dtype] = Dtype(4, "float", np.float32)
  @staticmethod
  def from_np(x) -> Dtype: return DTYPES_DICT[np.dtype(x).name]

DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}