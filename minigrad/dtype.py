from typing import Optional, Final
import numpy as np

class Dtype:
  itemsize: int
  name: str
  np = Optional[type]
  def __repr__(self): return f"dtypes.{self.name}"

class dtypes:
  float32: Final[Dtype] = Dtype(4, "float", np.float32)