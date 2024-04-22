from typing import Optional

"""
represents the device on which the tensor will be allocated
only support cpu for now
"""

class Device:
  DEFAULT = "CPU"
  @staticmethod
  def canonicalize(device: Optional[str]) -> str: return "CPU"