import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext; torchtext.disable_torchtext_deprecation_warning()
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
# Load the training set to build the vocabulary.
# Note: The new torchtext API returns iterators.
print("Building vocabulary from training data... (This may take a moment on first run)")
train_iter_for_vocab = IMDB(split='train')

def yield_tokens(data_iter):
  for _, text in data_iter:
      yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter_for_vocab), specials=["<unk>", "<pad>", "<cls>"], max_tokens=VOCAB_SIZE)
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
# The IMDB dataset from torchtext provides integer labels (e.g., 1 for positive, 2 for negative).
label_pipeline = lambda x: 1 if int(x) == 1 else 0 # Map 1 to 1 (pos), 2 to 0 (neg)

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
  This version is designed for `batch_first=True` inputs.
  """
  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x is expected to be of shape (batch_size, seq_len, d_model)
    x = x + self.pe[:x.size(1)]
    return self.dropout(x)

class TransformerClassifier(nn.Module):
  def __init__(self, vocab_size: int, embed_dim: int, nhead: int, d_hid: int,
                nlayers: int, dropout: float = 0.5):
    super().__init__()
    self.pos_encoder = PositionalEncoding(embed_dim, dropout)
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
    # Stack the encoder layers
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
    self.d_model = embed_dim
    # Final classification head
    self.classifier = nn.Linear(embed_dim, 2) # 2 classes: positive and negative

    self.init_weights()

  def init_weights(self) -> None:
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.classifier.bias.data.zero_()
    self.classifier.weight.data.uniform_(-initrange, initrange)

  def forward(self, src: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
    src = self.embedding(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src)
    # Pass through the transformer encoder
    output = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)
    
    # We use the output of the [CLS] token for classification
    # The [CLS] token is always at the first position
    cls_output = output[:, 0, :]
    
    # Pass through the final classification head
    return self.classifier(cls_output)

def train_one_epoch(model, dataloader, criterion, optimizer):
  model.train()
  total_acc, total_loss, total_count = 0, 0, 0
  log_interval = 100
  start_time = time.time()

  for idx, (text, labels) in enumerate(dataloader):
    optimizer.zero_grad()
    padding_mask = (text == PAD_IDX)
    predicted_labels = model(text, padding_mask)
    loss = criterion(predicted_labels, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    total_acc += (predicted_labels.argmax(1) == labels).sum().item()
    total_loss += loss.item()
    total_count += labels.size(0)
    if idx % log_interval == 0 and idx > 0:
      elapsed = time.time() - start_time
      print(f'| epoch {epoch:3d} | {idx:5d}/{len(dataloader):5d} batches | '
            f'accuracy {total_acc/total_count:.3f} | loss {loss.item():.4f}')
      total_acc, total_count = 0, 0
      start_time = time.time()

def evaluate(model, dataloader, criterion):
  model.eval()
  total_acc, total_loss, total_count = 0, 0, 0

  with torch.no_grad():
    for idx, (text, labels) in enumerate(dataloader):
      padding_mask = (text == PAD_IDX)
      predicted_labels = model(text, padding_mask)
      
      loss = criterion(predicted_labels, labels)
      
      total_acc += (predicted_labels.argmax(1) == labels).sum().item()
      total_loss += loss.item()
      total_count += labels.size(0)
          
  return total_acc / total_count, total_loss / len(dataloader)

if __name__ == '__main__':
  # Load the train and test sets. The iterators can only be consumed once.
  # We need to re-initialize them here for the DataLoaders.
  train_iter, test_iter = IMDB(split=('train', 'test'))
  # list() consumes the iterator and loads the data into memory.
  # This is acceptable for IMDB but not for very large datasets.
  train_dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
  test_dataloader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

  # Instantiate the model
  model = TransformerClassifier(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS).to(device)

  # Define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

  # Training loop
  for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    train_one_epoch(model, train_dataloader, criterion, optimizer)
    acc_val, loss_val = evaluate(model, test_dataloader, criterion)
    print('-' * 59)
    print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
          f'valid accuracy {acc_val:.3f} | valid loss {loss_val:.4f}')
    print('-' * 59)

  print("Training finished. Final test evaluation:")
  acc_test, loss_test = evaluate(model, test_dataloader, criterion)
  print(f'Test Accuracy: {acc_test:.3f}, Test Loss: {loss_test:.4f}')
