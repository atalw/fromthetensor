from tinygrad import Tensor, nn

hyp = {
  'epochs':               200,
  'opt': {
    'lr':                 0.005,
    'lr_decay':           0.98
  }
}

class Pos2Vec:
  def __init__(self, level):
    if level >= 1:
      self.level1 = [
        nn.Linear(773, 600),
        lambda x: x.relu(),
        nn.Linear(600, 773),
      ]
    if level >= 2:
      self.fc1 = nn.Linear(773, 600)
      self.level2 = [
        nn.Linear(600, 400),
        lambda x: x.relu(),
        nn.Linear(400, 600),
      ]
    if level >= 3:
      self.fc2 = nn.Linear(600, 400)
      self.level3 = [
        nn.Linear(400, 200),
        lambda x: x.relu(),
        nn.Linear(200, 400),
      ]
    if level >= 4:
      self.fc3 = nn.Linear(400, 200)
      self.level4 = [
        nn.Linear(200, 100),
        lambda x: x.relu(),
        nn.Linear(100, 200),
      ]
    if level >= 5:
      self.fc4 = nn.Linear(200, 100)

  # forward autoencode pass
  def __call__(self, x: Tensor, level) -> Tensor:
    if level == 1:
      return x.sequential(self.level1)
    elif level == 2:
      x = self.__call__(x, 1).relu()
      x = self.fc1(x).relu()
      return x.sequential(self.level2)
    elif level == 3:
      x = self.__call__(x, 2).relu()
      x = self.fc2(x).relu()
      return x.sequential(self.level3)
    elif level == 4:
      x = self.__call__(x, 3).relu()
      x = self.fc3(x).relu()
      return x.sequential(self.level4)
    elif level == 5:
      x = self.__call__(x, 4).relu()
      x = self.fc4(x)
      return x
  
  # returns expected output for each level (without autoencode)
  # used for calculating loss
  def expected_output(self, x: Tensor, level) -> Tensor:
    if level == 1:
      return x
    elif level == 2:
      x = self.expected_output(x, 1)
      return self.fc1(x)
    elif level == 3:
      x = self.expected_output(x, 2)
      return self.fc2(x)
    elif level == 4:
      x = self.expected_output(x, 3)
      return self.fc3(x)
    elif level == 5:
      x = self.expected_output(x, 4)
      return self.fc4(x)

