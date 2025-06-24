import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
import copy
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

def clones(module, N):
  "Produce N identical layers."
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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

# The annoted transformer: https://nlp.seas.harvard.edu/annotated-transformer/
class Encoder(nn.Module):
  "Core encoder is a stack of N layers"
  def __init__(self, layer, N):
    super(Encoder, self).__init__()
    self.layers = clones(layer, N)
    self.norm = LayerNorm(layer.size)

  def forward(self, x, mask):
    "Pass the input (and mask) through each layer in turn."
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)

class LayerNorm(nn.Module):
  "Construct a layernorm module (See citation for details)."
  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
  """
  A residual connection followed by a layer norm.
  Note for code simplicity the norm is first as opposed to last.
  """

  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    "Apply residual connection to any sublayer with the same size."
    return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
  "Encoder is made up of self-attn and feed forward (defined below)"
  def __init__(self, size, self_attn, feed_forward, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn = self_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SublayerConnection(size, dropout), 2)
    self.size = size

  def forward(self, x, mask):
    "Follow Figure 1 (left) for connections."
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
    return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # The mask is True for positions we want to ignore (pads).
        # masked_fill fills elements with -1e9 where the mask is True.
        scores = scores.masked_fill(mask, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    "Take in model size and number of heads."
    super(MultiHeadedAttention, self).__init__()
    assert d_model % h == 0
    # We assume d_v always equals d_k
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    "Implements Figure 2"
    if mask is not None:
        # Same mask applied to all h heads.
        # For a padding mask of shape (N, S), we need to make it (N, 1, 1, S)
        # to broadcast correctly to the scores matrix of shape (N, h, S, S).
        mask = mask.unsqueeze(1).unsqueeze(2)
    nbatches = query.size(0)

    # 1) Do all the linear projections in batch from d_model => h x d_k
    query, key, value = [
        lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        for lin, x in zip(self.linears, (query, key, value))
    ]

    # 2) Apply attention on all the projected vectors in batch.
    x, self.attn = attention(
        query, key, value, mask=mask, dropout=self.dropout
    )

    # 3) "Concat" using a view and apply a final linear.
    x = (
        x.transpose(1, 2)
        .contiguous()
        .view(nbatches, -1, self.h * self.d_k)
    )
    del query
    del key
    del value
    return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
  "Implements FFN equation."
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = nn.Linear(d_model, d_ff)
    self.w_2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.w_2(self.dropout(self.relu(self.w_1(x))))


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
    # i = torch.arange(0, d_model, 2)
    # denominator = torch.pow(10000, i / d_model)
    # div_term = 1.0 / denominator
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term) # even dimensions
    pe[:, 1::2] = torch.cos(position * div_term) # odd dimensions
    self.register_buffer('pe', pe) # this means it's not a model parameter but save it with save_state and to load buffer to device. no backprop will be applied despite having a forward func.

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x is expected to be of shape (batch_size, seq_len, d_model)
    x = x + self.pe[:x.size(1)]
    return self.dropout(x)

class Transformer(nn.Module):
  def __init__(self, vocab_size, d_model, nhead, d_hid, nlayers, dropout=0.1):
    super(Transformer, self).__init__()
    c = copy.deepcopy
    attn = MultiHeadedAttention(nhead, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_hid, dropout)
    position = PositionalEncoding(d_model, dropout, max_len=MAX_LEN)

    # Build the full encoder by creating a template layer
    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    self.encoder = Encoder(encoder_layer, nlayers)

    # Embedding layer
    self.embedding = nn.Embedding(vocab_size, d_model)
    # Positional encoding
    self.pos_encoder = position
    # Final classification head
    self.classifier = nn.Linear(d_model, 2) # 2 classes: positive and negative

    self.d_model = d_model
    self.init_weights()

  def init_weights(self) -> None:
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.classifier.bias.data.zero_()
    self.classifier.weight.data.uniform_(-initrange, initrange)

  def forward(self, src: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
    # 1. Get embeddings and apply positional encoding
    src = self.embedding(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src)
    # 2. Pass through the transformer encoder
    output = self.encoder(src, src_pad_mask)
    # 3. We use the output of the [CLS] token for classification
    cls_output = output[:, 0, :]
    # 4. Pass through the final classification head
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

def predict_sentiment(model, text: str):
  model.eval()  # Set the model to evaluation mode
  
  # Preprocess the text using the same pipeline as training
  processed_text = text_pipeline(text)
  processed_text = processed_text[:MAX_LEN - 1] # Truncate
  processed_text = [CLS_IDX] + processed_text
  
  # Convert to tensor, add a batch dimension, and move to the correct device
  text_tensor = torch.tensor(processed_text, dtype=torch.int64).unsqueeze(0).to(device)
  
  # Create a padding mask (all False for a single, unpadded sequence)
  padding_mask = (text_tensor == PAD_IDX)
  
  with torch.no_grad():
    prediction = model(text_tensor, padding_mask)
  return "Positive" if prediction.argmax(1).item() == 0 else "Negative"

if __name__ == '__main__':
  # Load the train and test sets. The iterators can only be consumed once.
  # We need to re-initialize them here for the DataLoaders.
  train_iter, test_iter = IMDB(split=('train', 'test'))
  # list() consumes the iterator and loads the data into memory.
  # This is acceptable for IMDB but not for very large datasets.
  train_dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
  test_dataloader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

  # Instantiate the model
  model = Transformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS).to(device)

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

    # Example usage for prediction
  print("\n--- Predicting on custom reviews ---")
  review1 = "I absolutely loved this movie! The acting was superb and the plot was gripping."
  print(f"Review: '{review1}'")
  print(f"Predicted Sentiment: {predict_sentiment(model, review1)}\n")

  review2 = "This was a complete waste of time. The story was predictable and the characters were one-dimensional."
  print(f"Review: '{review2}'")
  print(f"Predicted Sentiment: {predict_sentiment(model, review2)}")