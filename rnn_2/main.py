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

def train(model, optimizer, criterion, input_tensor, target_tensor):
  hidden = model.init_hidden()
  optimizer.zero_grad()
  loss = 0
  
  # Loop through each character in the input chunk
  for i in range(input_tensor.size(0)):
    output, hidden = model(input_tensor[i], hidden)
    loss += criterion(output, target_tensor[i].unsqueeze(0))
      
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Gradient clipping
  optimizer.step()
  
  return loss.item() / input_tensor.size(0) # Return average loss

def generate(model, vocab_size, prime_str='X:9\nT:Generated\nM:3/8\nL:1/8\nR:Bourree (3 time)\nK:G\n', predict_len=200, temperature=0.8):
  model.eval() # Set model to evaluation mode
  
  with torch.no_grad():
    hidden = model.init_hidden()
    prime_input = char_to_tensor(prime_str, vocab_size)
    generated_text = prime_str

    # "Warm up" the hidden state with the prime string
    for i in range(len(prime_str) - 1):
      _, hidden = model(prime_input[i], hidden)
    
    # Use the last character of the prime string as the first input for generation
    current_input = prime_input[-1]

    for i in range(predict_len):
      output, hidden = model(current_input, hidden)
      
      # Apply temperature to the output probabilities
      output_dist = output.data.view(-1).div(temperature).exp()
      top_i = torch.multinomial(output_dist, 1)[0]
      
      # Append the predicted character and update the input for the next step
      predicted_char = idx_to_char[top_i.item()]
      generated_text += predicted_char
      current_input = char_to_tensor(predicted_char, vocab_size)[0]
          
  model.train() # Set model back to training mode
  return generated_text

if __name__ == "__main__":
  data = read_music()
  vocab_size, char_to_idx, idx_to_char = create_music_vocabulary(data)
  input_chunk, target_chunk = get_random_chunk(data, vocab_size)

  hidden_size = 128
  learning_rate = 0.005
  n_iters = 2000
  print_interval = 100

  rnn = SimpleRNN(vocab_size, hidden_size, vocab_size)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

  all_losses = []
  current_loss = 0
  st = time.time()

  for i in range(1, n_iters + 1):
      input_chunk, target_chunk = get_random_chunk(data, vocab_size)
      loss = train(rnn, optimizer, criterion, input_chunk, target_chunk)
      current_loss += loss
      
      if i % print_interval == 0:
          elapsed = time.time() - st 
          avg_loss = current_loss / print_interval 
          all_losses.append(avg_loss)
          current_loss = 0
          
          print(f"Iteration: {i}/{n_iters} ({i/n_iters*100:.0f}%) | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {elapsed:.2f}s")
          
          print("--- Generated Sample ---")
          print(generate(rnn, vocab_size))
          print("------------------------\n")

  print("Training finished.")
  print("\n--- Final Generated Music ---")
  print(generate(rnn, vocab_size, predict_len=500))
  print("-----------------------------\n")
