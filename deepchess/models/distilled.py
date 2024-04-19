from tinygrad import Tensor, nn

hyp = {
  'epochs':               1000,
  'opt': {
    'lr':                 0.01,
    'lr_decay':           0.99
  }
}


class Distilled:
  def __init__(self):
    self.layers1 = [
      nn.Linear(773*2, 100*2),
      lambda x: x.relu(),
      nn.Linear(100*2, 100*2),
      lambda x: x.relu(),
      nn.Linear(100*2, 100*2),
      lambda x: x.relu(),
      nn.Linear(100*2, 100*2),
      lambda x: x.relu(),
    ]
    self.layers2 = [
      nn.Linear(100*2, 100),
      lambda x: x.relu(),
      nn.Linear(100, 100),
      lambda x: x.relu(),
      nn.Linear(100, 2),
      lambda x: x.relu(),
    ]

  def encode(self, x: Tensor) -> Tensor:
    return x.sequential(self.layers1)
  
  def __call__(self, x: Tensor) -> Tensor:
    return self.encode(x).sequential(self.layers2)