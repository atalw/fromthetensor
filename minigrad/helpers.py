from typing import Iterable, Union
import functools
import operator

def prod(x:Iterable) -> int: return functools.reduce(operator.mul, x, 1)