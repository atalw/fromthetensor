import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math
import os

def read_music():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(current_dir, "french_music.txt")
  try:
    with open(file_path, 'r', encoding='latin-1') as f:
      music_data = f.read()
      return music_data
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return [], 0
  except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    return [], 0

def create_music_vocabulary(data):
  all_characters = sorted(list(set(data)))
  vocab_size = len(all_characters)
  print(f"Vocabulary contains {vocab_size} unique characters.")
  print(f"Vocabulary: {''.join(all_characters)}")
  # Create character-to-index and index-to-character mappings
  char_to_idx = {char: i for i, char in enumerate(all_characters)}
  idx_to_char = {i: char for i, char in enumerate(all_characters)}
  return vocab_size, char_to_idx, idx_to_char

def char_to_tensor(s, vocab_size):
  tensor = torch.zeros(len(s), 1, vocab_size)
  for i, char in enumerate(s):
      tensor[i][0][char_to_idx[char]] = 1
  return tensor

def get_random_chunk(data, vocab_size):
  chunk_len = 100 # How many characters to train on at a time
  start_index = random.randint(0, len(data) - chunk_len)
  end_index = start_index + chunk_len + 1
  chunk = data[start_index:end_index]
  
  # The input is all characters except the last
  input_chunk = char_to_tensor(chunk[:-1], vocab_size)
  # The target is all characters except the first
  target_chunk = torch.tensor([char_to_idx[c] for c in chunk[1:]], dtype=torch.long)
  
  return input_chunk, target_chunk

class SimpleRNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleRNN, self).__init__()
    self.hidden_size = hidden_size
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(input_size + hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input_char, hidden_state):
    combined = torch.cat((input_char, hidden_state), 1)
    # This is the core recurrent equation: h_t = tanh(W_xh*x_t + W_hh*h_{t-1} + b_h)
    hidden = torch.tanh(self.i2h(combined))
    output = self.i2o(combined)
    output = self.softmax(output)
    return output, hidden

  def init_hidden(self):
    return torch.zeros(1, self.hidden_size)

if __name__ == "__main__":
  data = read_music()
  vocab_size, char_to_idx, idx_to_char = create_music_vocabulary(data)
  print(char_to_idx)
  print(idx_to_char)

  input_chunk, target_chunk = get_random_chunk(data, vocab_size)
  print(input_chunk)
  print(target_chunk)

