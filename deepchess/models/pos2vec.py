from tinygrad import Tensor, nn

hyp = {
  'epochs':               200,
  'opt': {
    'lr':                 0.005,
    'lr_decay':           0.98
  }
}

class Pos2Vec:
  def __init__(self):
    self.encode_layers = [
      nn.Linear(773, 600),
      lambda x: x.leakyrelu(),
      nn.Linear(600, 400),
      lambda x: x.leakyrelu(),
      nn.Linear(400, 200),
      lambda x: x.leakyrelu(),
      nn.Linear(200, 100),
    ]

    self.decode_layers = [
      nn.Linear(100, 200),
      lambda x: x.leakyrelu(),
      nn.Linear(200, 400),
      lambda x: x.leakyrelu(),
      nn.Linear(400, 600),
      lambda x: x.leakyrelu(),
      nn.Linear(600, 773),
    ]
  
  def encode(self, x):
    return x.sequential(self.encode_layers)
  
  def decode(self, x):
    return x.sequential(self.decode_layers)

  # forward autoencode pass
  def __call__(self, x: Tensor, level=5) -> Tensor:
    enc = self.encode(x)
    return self.decode(enc)