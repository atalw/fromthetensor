from typing import Iterable, Union, Tuple, Iterator
import functools
import operator

def prod(x:Iterable) -> int: return functools.reduce(operator.mul, x, 1)
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def dedup(x): return list(dict.fromkeys(x))   # retains list order
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]