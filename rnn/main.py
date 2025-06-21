import os
import random
from string import ascii_letters
from unidecode import unidecode

from tinygrad import nn, dtypes, Tensor
from tinygrad.nn.state import get_parameters

data_dir = "./data/names"
lang2label = {
    file_name.split(".")[0]: Tensor([i], dtype=dtypes.long)
    for i, file_name in enumerate(os.listdir(data_dir))
}

char2idx = {letter: i for i, letter in enumerate(ascii_letters + " .,:;-'")}
num_langs = len(lang2label)
num_letters = len(char2idx)

def name_to_tensor(name):
  # In PyTorch, RNN layers expect the input tensor to be of size (seq_len, batch_size, input_size)
  t = Tensor.zeros(len(name), 1, num_letters)
  t = t.contiguous() # Explicitly make the tensor contiguous
  for i, char in enumerate(name):
    t[i, 0, char2idx[char]] = 1
  return t 

# name_to_tensor and name_to_tensor_one_hot produce the same result
def name_to_tensor_one_hot(name):
  idxs = [char2idx[char] for char in name]
  t = Tensor.one_hot(Tensor(idxs), num_letters).cast(dtypes.float32)
  return t.reshape(len(name), 1, num_letters)

def create_dataset():
  tensor_names = []
  target_langs = []

  for file in os.listdir(data_dir):
    with open(os.path.join(data_dir, file)) as f:
      lang = file.split(".")[0]
      names = [unidecode(line.rstrip()) for line in f]
      for name in names:
        try:
          tensor_names.append(name_to_tensor_one_hot(name))
          target_langs.append(lang2label[lang])
        except KeyError:
          pass
  train_idx = int(len(target_langs)*0.9)
  train_dataset = list(zip(tensor_names[:train_idx], target_langs[:train_idx]))
  test_dataset = list(zip(tensor_names[train_idx:], target_langs[train_idx:]))
  return train_dataset, test_dataset

class RNN():
  def __init__(self, input_size, hidden_size, output_size):
    self.hidden_size = hidden_size
    self.in_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
    self.in_to_output = nn.Linear(input_size + hidden_size, output_size)

  def __call__(self, x, hidden):
    combined = x.cat(hidden, dim=1)
    hidden = Tensor.sigmoid(self.in_to_hidden(combined))
    output = self.in_to_output(combined)
    return output, hidden

  def init_hidden(self):
    return Tensor.kaiming_uniform(1, self.hidden_size)

def train(train_dataset):
  with Tensor.train():
    hidden_size = 64 
    learning_rate = 0.001

    model = RNN(num_letters, hidden_size, num_langs)
    opt = nn.optim.Adam(get_parameters(model), learning_rate)

    num_epochs = 2
    print_interval = 1000

    # Revisit when tinygrad releases 0.10.4 as it will have https://github.com/tinygrad/tinygrad/pull/10510 upstreamed. Buffer limit of 30 kernels on METAL is blocking training.
    for epoch in range(num_epochs):
      random.shuffle(train_dataset)
      for i, (name, label) in enumerate(train_dataset):
        hidden = model.init_hidden()
        for char in name:
          output, hidden = model(char, hidden)
        loss = output.sparse_categorical_crossentropy(label)
        opt.zero_grad()
        loss.backward()
        for param in get_parameters(model):
          param.grad = param.grad.clip(-1, 1)
        opt.step()

        if (i + 1) % print_interval == 0:
          print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Step [{i + 1}/{len(train_dataset)}], "
            f"Loss: {loss.item():.4f}"
          )

if __name__ == "__main__":
  # print(name_to_tensor("abc").numpy())
  # print(name_to_tensor_one_hot("abc").numpy())
  train_dataset, test_dataset = create_dataset()
  print("Train: ", len(train_dataset))
  print("Test: ", len(test_dataset))

  train(train_dataset)
