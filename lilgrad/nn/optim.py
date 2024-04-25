from typing import List 
from tensor import Tensor
from helpers import dedup

class Optimizer:
  def __init__(self, params: List[Tensor], lr: float):
    for p in params:
      if p.requires_grad is None: p.requires_grad = True
    
    self.params = dedup([p for p in params if p.requires_grad])
    assert len(self.params) != 0, "optimizer must have at least 1 param"
    self.device = self.params[0].device
    self.buffers: List[Tensor] = dedup([x for x in params if not x.requires_grad])
    self.lr = Tensor([lr], requires_grad=False, device=self.device)
  
  def zero_grad(self):
    for param in self.params: param.grad = None
  
class SGD(Optimizer):
  def __init__(self, params: List[Tensor], lr=0.001, momentum=0, weight_decay=0.0, nesterov=False):
    super().__init__(params, lr)
    self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
    self.b = [Tensor.zeros(*t.shape, requires_grad=False, device=t.device) for t in self.params] if self.momentum else []

  # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
  def step(self):
    for i, t in enumerate(self.params):
      assert t.grad is not None
      g = t.grad + self.wd * t.detach()
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g) # NOTE: self.b[i] is 0 on first run, no if required
        g = g + self.momentum * self.b[i] if self.nesterov else self.b[i]
      t.assign(t.detach() - g * self.lr)
      

"""
nn optimizers

SGD
Adam

"""