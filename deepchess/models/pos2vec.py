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
    self.layer0 = [
      nn.Linear(773, 600),
      lambda x: x.relu(),
      nn.Linear(600, 600),
      lambda x: x.relu(),
      nn.Linear(600, 773),
      lambda x: x.relu(),
    ]
    self.fc0 = nn.Linear(773, 600)
    self.layer1 = [
      nn.Linear(600, 400),
      lambda x: x.relu(),
      nn.Linear(400, 400),
      lambda x: x.relu(),
      nn.Linear(400, 600),
      lambda x: x.relu(),
    ]
    self.fc1 = nn.Linear(600, 400)
    self.layer2 = [
      nn.Linear(400, 200),
      lambda x: x.relu(),
      nn.Linear(200, 200),
      lambda x: x.relu(),
      nn.Linear(200, 400),
      lambda x: x.relu(),
    ]
    self.fc2 = nn.Linear(400, 200)
    self.layer3 = [
      nn.Linear(200, 100),
      lambda x: x.relu(),
      nn.Linear(100, 100),
      lambda x: x.relu(),
      nn.Linear(100, 200),
      lambda x: x.relu(),
    ]
    self.fc3 = nn.Linear(200, 100)
  
  def encode(self, x: Tensor, layer=4) -> Tensor:
    if layer == 0:
      x = x.sequential(self.layer0)
      return self.fc0(x).relu()
    elif layer == 1:
      x = self.encode(x, layer-1)
      x = x.sequential(self.layer1)
      return self.fc1(x).relu()
    elif layer == 2:
      x = self.encode(x, layer-1)
      x = x.sequential(self.layer2)
      return self.fc2(x).relu()
    elif layer == 3:
      x = self.encode(x, layer-1)
      x = x.sequential(self.layer3)
      return self.fc3(x).relu()
    elif layer == 4:
      return self.encode(x, layer-1)
    else:
      raise NotImplementedError()
  
  def __call__(self, x:Tensor, layer=4) -> Tensor:
    if layer == 0:
      return x.sequential(self.layer0)
    elif layer == 1:
      x = self.encode(x, layer-1)
      return x.sequential(self.layer1)
    elif layer == 2:
      x = self.encode(x, layer-1)
      return x.sequential(self.layer2)
    elif layer == 3:
      x = self.encode(x, layer-1)
      return x.sequential(self.layer3)
    elif layer == 4:
      return self.encode(x, layer)
    else:
      raise NotImplementedError()