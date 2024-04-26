from typing import Dict, List
from tensor import Tensor

def get_state_dict(obj) -> Dict[str, Tensor]:
  if hasattr(obj, '_asdict'): return get_state_dict(obj._asdict())  # namedtuple
  if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__)
  state_dict = {}
  if isinstance(obj, (list, tuple)):
    for i,x in enumerate(obj): state_dict.update(get_state_dict(x, f"{str(i)}."))
  elif isinstance(obj, dict):
    for k,v in obj.items(): state_dict.update(get_state_dict(v, f"{str(k)}."))
  return state_dict

def get_parameters(obj) -> List[Tensor]: return list(get_state_dict(obj).values())