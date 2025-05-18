import os
import random
from string import ascii_letters
from unidecode import unidecode

from tinygrad import nn, dtypes, Tensor

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

if __name__ == "__main__":
  # print(name_to_tensor("abc").numpy())
  # print(name_to_tensor_one_hot("abc").numpy())
  train_dataset, test_dataset = create_dataset()
  print("Train: ", len(train_dataset))
  print("Test: ", len(test_dataset))
