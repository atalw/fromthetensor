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
    self.layers = [
      nn.Linear(773*2, 100*2),
      nn.Linear(100*2, 100),
      nn.Linear(100, 100),
      nn.Linear(100, 100),

      nn.Linear(100, 100),
      nn.Linear(100, 100),
      nn.Linear(100, 2),
      lambda x: x.sigmoid(),
    ]

  
  def __call__(self, x: Tensor) -> Tensor:
    return x.sequential(self.layers)