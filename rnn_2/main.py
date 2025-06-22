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

def char_to_tensor(s):
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
  input_chunk = char_to_tensor(chunk[:-1])
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

# https://arxiv.org/pdf/1412.3555
class GRU_RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(GRU_RNN, self).__init__()
    self.hidden_size = hidden_size
    self.gru = nn.GRU(input_size, hidden_size)
    self.h2o = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input_char, hidden_state):
    # nn.GRU expects input shape (seq_len, batch, input_size)
    # Our input is (1, 1, vocab_size) which is correct for a single step
    output, hidden = self.gru(input_char, hidden_state)
    output = self.h2o(output.view(1, -1))
    output = self.softmax(output)
    return output, hidden

  def init_hidden(self):
    # The hidden state for a GRU is a single tensor
    return torch.zeros(1, 1, self.hidden_size)

# Model 3: LSTM-based RNN
# This model uses the Long Short-Term Memory (LSTM) unit.
class LSTM_RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(LSTM_RNN, self).__init__()
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size, hidden_size)
    self.h2o = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input_char, hidden_cell_tuple):
    # nn.LSTM expects input shape (seq_len, batch, input_size)
    output, (hidden, cell) = self.lstm(input_char, hidden_cell_tuple)
    output = self.h2o(output.view(1, -1))
    output = self.softmax(output)
    return output, (hidden, cell)

  def init_hidden(self):
    # The hidden state for an LSTM is a tuple containing the hidden state and the cell state
    hidden_state = torch.zeros(1, 1, self.hidden_size)
    cell_state = torch.zeros(1, 1, self.hidden_size)
    return (hidden_state, cell_state)


def train(model, optimizer, criterion, input_tensor, target_tensor):
  hidden = model.init_hidden()
  optimizer.zero_grad()
  loss = 0
  
  # Loop through each character in the input chunk
  for i in range(input_tensor.size(0)):
    current_input_for_model = input_tensor[i] # Shape (1, vocab_size)
    if isinstance(model, (GRU_RNN, LSTM_RNN)):
        # For GRU/LSTM, input needs to be (seq_len, batch, input_size)
        # Since we're processing one char at a time, seq_len=1, batch=1
        current_input_for_model = current_input_for_model.unsqueeze(0) # Shape (1, 1, vocab_size)
    output, hidden = model(current_input_for_model, hidden)
    loss += criterion(output, target_tensor[i].unsqueeze(0))
      
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Gradient clipping
  optimizer.step()
  
  return loss.item() / input_tensor.size(0) # Return average loss

def generate(model, prime_str='X:9\nT:Generated\nM:3/8\nL:1/8\nR:Bourree (3 time)\nK:G\n', predict_len=200, temperature=0.8):
  model.eval() # Set model to evaluation mode
  
  with torch.no_grad():
    hidden = model.init_hidden()
    generated_text = prime_str

    # "Warm up" the hidden state with the prime string
    for char in prime_str:
      char_input = char_to_tensor(char)
      current_input = char_input[0] if isinstance(model, SimpleRNN) else char_input[0].unsqueeze(0)
      _, hidden = model(current_input, hidden)
    
    # Use the last character of the prime string as the first input for generation
    current_char = prime_str[-1]

    for i in range(predict_len):
      char_input = char_to_tensor(current_char)
      current_input = char_input[0] if isinstance(model, SimpleRNN) else char_input[0].unsqueeze(0)
      output, hidden = model(current_input, hidden)
      
      # Apply temperature to the output probabilities
      output_dist = output.data.view(-1).div(temperature).exp()
      top_i = torch.multinomial(output_dist, 1)[0]
      
      # Append the predicted character and update the input for the next step
      predicted_char = idx_to_char[top_i.item()]
      generated_text += predicted_char
      current_char = predicted_char
          
  model.train() # Set model back to training mode
  return generated_text

def run_experiment(model_type, n_iters=2000):
  print(f"\n===== Running Experiment for: {model_type.__name__} =====")

  
  # Hyperparameters
  hidden_size = 128
  learning_rate = 0.005
  
  # Initialize model, loss, and optimizer
  model = model_type(vocab_size, hidden_size, vocab_size)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
  start_time = time.time()
  all_losses = []
  
  for i in range(1, n_iters + 1):
      input_chunk, target_chunk = get_random_chunk(data, vocab_size)
      loss = train(model, optimizer, criterion, input_chunk, target_chunk)
      all_losses.append(loss)
      
  end_time = time.time()
  training_time = end_time - start_time
  final_loss = sum(all_losses[-100:]) / 100 # Avg of last 100 losses
  
  print(f"Training finished in {training_time:.2f} seconds.")
  print(f"Final average loss: {final_loss:.4f}")
  
  print("\n--- Generated Sample ---")
  generated_sample = generate(model)
  print(generated_sample)
  print("------------------------\n")
  
  return {
      'model_name': model_type.__name__,
      'time': training_time,
      'loss': final_loss,
      'sample': generated_sample
  }

if __name__ == "__main__":
  data = read_music()
  vocab_size, char_to_idx, idx_to_char = create_music_vocabulary(data)

  results = []
  # Run experiment for all three models
  results.append(run_experiment(SimpleRNN))
  results.append(run_experiment(GRU_RNN))
  results.append(run_experiment(LSTM_RNN))
  
  # --- 6. Final Comparison ---
  print("\n\n===== EMPIRICAL EVALUATION SUMMARY =====")
  print(f"{'Model':<12} | {'Training Time (s)':<20} | {'Final Avg Loss':<18}")
  print("-" * 55)
  for res in results:
      print(f"{res['model_name']:<12} | {res['time']:<20.2f} | {res['loss']:<18.4f}")
  print("-" * 55)