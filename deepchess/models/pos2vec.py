from tinygrad import Tensor, nn

hyp = {
  'epochs':               200,
  'opt': {
    'lr':                 0.005,
    'lr_decay_decay':     0.98
  }
}

class Pos2Vec:
  def __init__(self):
    self.layers = [
      nn.Linear(773, 600),
      lambda x: x.relu(),
      nn.Linear(600, 773),
      lambda x: x.relu(),

      nn.Linear(773, 600),

      nn.Linear(600, 400),
      lambda x: x.relu(),
      nn.Linear(400, 600),
      lambda x: x.relu(),

      nn.Linear(600, 400),

      nn.Linear(400, 200),
      lambda x: x.relu(),
      nn.Linear(200, 400),
      lambda x: x.relu(),

      nn.Linear(400, 200),

      nn.Linear(200, 100),
      lambda x: x.relu(),
      nn.Linear(100, 200),
    ]

  
  def __call__(self, x: Tensor) -> Tensor:
    return x.sequential(self.layers)