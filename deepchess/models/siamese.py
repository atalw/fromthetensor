from tinygrad import Tensor, nn

hyp = {
  'epochs':               1000,
  'opt': {
    'lr':                 0.01,
    'lr_decay':           0.99
  }
}


class Siamese:
  def __init__(self):
    self.layers = [
      nn.Linear(200, 400),
      lambda x: x.leakyrelu(),
      nn.Linear(400, 200),
      lambda x: x.leakyrelu(),
      nn.Linear(200, 100),
      lambda x: x.leakyrelu(),
      nn.Linear(100, 2),
      lambda x: x.sigmoid(),
    ]

  
  def __call__(self, x: Tensor) -> Tensor:
    return x.sequential(self.layers)