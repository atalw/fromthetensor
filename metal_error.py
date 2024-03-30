# https://discord.com/channels/1068976834382925865/1069001075828469790/1189831398975094804
from tinygrad import Tensor

# force a long loop in subsequent kernels. seems to start failing at ~1e6 for me but gets consistent as we increase
N = int(1e8) + 1
while 1:
    ones = Tensor.ones((N,)).contiguous()

    a = ones.mean().item()
    print(f"{a=}") # ends up being 0.0 on gpu backends... possible silent metal fail?

    intermediate = Tensor(3.14159).item() # eval any tensor with same shape
    print(f"{intermediate=}")

    b = ones.mean().item()
    print(f"{b=}") # gets same val as intermediate when LRUAllocator is enabled

    assert a == b
    break
