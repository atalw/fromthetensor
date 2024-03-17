from tinygrad.tensor import Tensor
from tinygrad.nn import Linear

class IrisModel:
    def __init__(self):
        self.l1 = Linear(4, 16)
        self.l2 = Linear(16, 3)

    def __call__(self, x):
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x