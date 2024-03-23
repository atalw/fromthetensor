from models.batchnorm import BatchNorm1d
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
      BatchNorm1d(400),
      lambda x: x.relu(),
      nn.Linear(400, 200),
      BatchNorm1d(200),
      lambda x: x.relu(),
      nn.Linear(200, 100),
      BatchNorm1d(100),
      lambda x: x.relu(),
      nn.Linear(100, 2),
      BatchNorm1d(2),
      # lambda x: x.relu(),
      lambda x: x.sigmoid(),
    ]

  
  def __call__(self, x: Tensor) -> Tensor:
    return x.sequential(self.layers)