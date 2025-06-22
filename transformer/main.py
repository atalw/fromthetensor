import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
import time

# Set the device (MPS for Apple Silicon, CUDA for Nvidia, or CPU)
if torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cpu")

print(f"Using device: {device}")

# Model Hyperparameters
VOCAB_SIZE = 10000  # Size of the vocabulary
EMBED_DIM = 256     # Embedding dimension
HIDDEN_DIM = 512    # Dimension of the feedforward network model in nn.TransformerEncoder
NUM_HEADS = 8       # Number of heads in the multi-head attention models
NUM_LAYERS = 3      # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
MAX_LEN = 256       # Maximum sequence length

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 3

tokenizer = get_tokenizer('basic_english')
train_iter, test_iter = IMDB()

def yield_tokens(data_iter):
  for _, text in data_iter:
      yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>", "<cls>"], max_tokens=VOCAB_SIZE)
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0

PAD_IDX = vocab['<pad>']
CLS_IDX = vocab['<cls>']

def collate_batch(batch):
  """
  Collates a batch of data. Adds <cls> token, truncates, pads, and creates tensors.
  This function is passed to the DataLoader.
  """
  label_list, text_list = [], []
  for (_label, _text) in batch:
    # Process label
    label_list.append(label_pipeline(_label))
    
    # Process text: add <cls>, truncate, and convert to IDs
    processed_text = text_pipeline(_text)
    processed_text = processed_text[:MAX_LEN - 1] # Truncate, leave space for <cls>
    processed_text = [CLS_IDX] + processed_text
    text_list.append(torch.tensor(processed_text, dtype=torch.int64))
      
  # Pad all sequences in the batch to the same length
  padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
  
  # Convert labels to a tensor
  label_list = torch.tensor(label_list, dtype=torch.int64)
  return padded_text_list.to(device), label_list.to(device)

class PositionalEncoding(nn.Module):
  """
  Injects some information about the relative or absolute position of the tokens in the sequence.
  The positional encodings have the same dimension as the embeddings so that the two can be summed.
  """
  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    # i = torch.arange(0, d_model, 2)
    # denominator = torch.pow(10000, i / d_model)
    # div_term = 1.0 / denominator
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term) # even dimensions
    pe[:, 0, 1::2] = torch.cos(position * div_term) # odd dimensions
    self.register_buffer('pe', pe) # this means it's not a model parameter but save it with save_state and to load buffer to device. no backprop will be applied despite having a forward func.

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)


