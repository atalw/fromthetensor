from typing import Iterable
import functools
import operator

def prod(x:Iterable) -> int: return functools.reduce(operator.mul, x, 1)
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x