from typing import Iterable, Union, Tuple, Iterator, Optional
import functools
import operator
from tqdm import tqdm
import os, platform, pathlib, hashlib, urllib.request, tempfile

def prod(x:Iterable) -> int: return functools.reduce(operator.mul, x, 1)
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def dedup(x): return list(dict.fromkeys(x))   # retains list order
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def round_up(num, amt:int): return (num+amt-1)//amt * amt

OSX = platform.system() == "Darwin"
_cache_dir: str = os.path.expanduser("~/Library/Caches" if OSX else "~/.cache")
CACHEDB: str = os.path.abspath(os.path.join(_cache_dir, "lilgrad", "cache.db"))

def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None) -> pathlib.Path:
  if url.startswith(("/", ".")): return pathlib.Path(url)
  fp = pathlib.Path(name) if name is not None and (isinstance(name, pathlib.Path) or '/' in name) else pathlib.Path(_cache_dir) / "lilgrad" / "downloads" / (name if name else hashlib.md5(url.encode('utf-8')).hexdigest())  # noqa: E501
  if not fp.is_file():
    with urllib.request.urlopen(url, timeout=10) as r:
      assert r.status == 200
      total_length = int(r.headers.get('content-length', 0))
      progress_bar = tqdm(total=total_length, unit='B', unit_scale=True, desc=url)
      (path := fp.parent).mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        while chunk := r.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        if (file_size:=os.stat(f.name).st_size) < total_length: raise RuntimeError(f"fetch size incomplete, {file_size} < {total_length}")
        pathlib.Path(f.name).rename(fp)
  return fp