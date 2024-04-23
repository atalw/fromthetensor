import time, math, unittest
import numpy as np
import torch
# from helpers import getenv
from tensor import Tensor
from device import Device
from dtype import dtypes

# FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
# PRINT_TENSORS = getenv("PRINT_TENSORS", 0)
FORWARD_ONLY = 0
PRINT_TENSORS = 0

def prepare_test_op(low, high, shps, vals, forward_only=False):
  if shps is None:
    ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
  else:
    np.random.seed(0)
    np_data = [np.random.uniform(low=low, high=high, size=size).astype(dtypes.float32.np) for size in shps]
    ts = [torch.tensor(data, requires_grad=(not forward_only)) for data in np_data]
  tst = [Tensor(x.detach().numpy(), requires_grad=(not forward_only and not FORWARD_ONLY)) for x in ts]
  return ts, tst

def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                   forward_only=False, vals=None, low=-1.5, high=1.5):
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  ts, tst = prepare_test_op(low, high, shps, vals, forward_only)

  st = time.monotonic()
  out = torch_fxn(*ts)
  torch_fp = time.monotonic() - st

  # move inputs to a different device, test the device of intermediate tensors are correct
  # if mt:=getenv("MOVE_TENSOR", ""):
    # for t in tst: t.to_(mt)

  st = time.monotonic()
  ret = tinygrad_fxn(*tst).realize()
  tinygrad_fp = time.monotonic() - st

  def compare(s, tinygrad_output, torch_output, atol, rtol):
    if PRINT_TENSORS: print(s, tinygrad_output, torch_output)
    try:
      assert tinygrad_output.shape == torch_output.shape, f"shape mismatch: tinygrad={tinygrad_output.shape} | torch={torch_output.shape}"
      assert tinygrad_output.dtype == torch_output.dtype, f"dtype mismatch: tinygrad={tinygrad_output.dtype} | torch={torch_output.dtype}"
      np.testing.assert_allclose(tinygrad_output, torch_output, atol=atol, rtol=rtol)
    except Exception as e:
      raise Exception(f"{s} failed shape {tinygrad_output.shape}: {e}")

  compare("forward pass", ret.numpy(), out.detach().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY:
    st = time.monotonic()
    (out+1).square().mean().backward()
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    (ret+1).square().mean().backward()
    for tt in tst: tt.grad.realize()
    tinygrad_fbp = time.monotonic() - st

    for i, (t, tt) in enumerate(zip(ts, tst)):
      compare(f"backward pass tensor {i}", tt.grad.numpy(), t.grad.detach().numpy(), atol=grad_atol, rtol=grad_rtol)

  print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % \
        (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")
  
class TestOps(unittest.TestCase):
  def test_gather(self):
    # indices cannot have gradient
    # indices cannot be negative (torch gather)
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op([(4,5,6)], lambda x: x.gather(index=b, dim=0), lambda x: x.gather(idx=a, dim=0))
    helper_test_op([(4,5,6)], lambda x: x.gather(index=b, dim=1), lambda x: x.gather(idx=a, dim=1))
    helper_test_op([(4,5,6)], lambda x: x.gather(index=b, dim=2), lambda x: x.gather(idx=a, dim=2))
    helper_test_op([(3,4,5)], lambda x: x.gather(index=b, dim=0), lambda x: x.gather(idx=a, dim=0))
    self.helper_test_exception([(4,5,6)], lambda x: x.gather(index=torch.tensor([1], dtype=torch.int64), dim=0),
                                          lambda x: x.gather(idx=Tensor([1], dtype=dtypes.int32), dim=0), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,1,1)], lambda x: x.gather(index=b, dim=0),
                                          lambda x: x.gather(idx=a, dim=0), expected=(RuntimeError, AssertionError))

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)