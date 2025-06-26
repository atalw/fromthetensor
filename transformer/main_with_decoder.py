import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
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
NUM_EPOCHS = 10

tokenizer = get_tokenizer('basic_english')
# Load the training set to build the vocabulary.
# Note: The new torchtext API returns iterators.
print("Building vocabulary from training data... (This may take a moment on first run)")
train_iter_for_vocab = IMDB(split='train')

def yield_tokens(data_iter):
  for _, text in data_iter:
      yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter_for_vocab), specials=["<unk>", "<pad>", "<sos>", "<eos>", "<positive>", "<negative>"], max_tokens=VOCAB_SIZE)
vocab.set_default_index(vocab["<unk>"])

PAD_IDX = vocab['<pad>']
SOS_IDX = vocab['<sos>']
EOS_IDX = vocab['<eos>']
POSITIVE_IDX = vocab['<positive>']
NEGATIVE_IDX = vocab['<negative>']

text_pipeline = lambda x: vocab(tokenizer(x))
# The IMDB dataset from torchtext provides integer labels (e.g., 1 for positive, 2 for negative).
label_pipeline = lambda x: POSITIVE_IDX if int(x) == 1 else NEGATIVE_IDX # Map 1 to POS_IDX, 2 to NEG_IDX

def clones(module, N):
  "Produce N identical layers."
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def collate_batch_generative(batch):
  """
  Collates a batch of data for the generative sequence-to-sequence task.
  """
  src_list, tgt_list = [], []
  for (_label, _text) in batch:
    # Source is the sentiment token
    src_list.append(torch.tensor([label_pipeline(_label)], dtype=torch.int64))
    # Target is the review text, truncated and with SOS/EOS tokens
    processed_text = text_pipeline(_text)
    processed_text = processed_text[:MAX_LEN - 2] # Truncate, leave space for SOS and EOS
    tgt_list.append(torch.tensor([SOS_IDX] + processed_text + [EOS_IDX], dtype=torch.int64))

  src_tensors = torch.cat(src_list).unsqueeze(1) # Shape: (batch_size, 1)
  tgt_tensors = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)
  return src_tensors.to(device), tgt_tensors.to(device)

# The annoted transformer: https://nlp.seas.harvard.edu/annotated-transformer/
class EncoderDecoder(nn.Module):
  """
  A standard Encoder-Decoder architecture. Base for this and many
  other models.
  """
  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator

  def forward(self, src, tgt, src_mask, tgt_mask):
    "Take in and process masked src and target sequences."
    return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

  def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)

  def decode(self, memory, src_mask, tgt, tgt_mask):
    return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
  "Define standard linear + softmax generation step."

  def __init__(self, d_model, vocab):
    super(Generator, self).__init__()
    self.proj = nn.Linear(d_model, vocab)

  def forward(self, x):
    return F.log_softmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
  "Core encoder is a stack of N layers"
  def __init__(self, layer, N):
    super(Encoder, self).__init__()
    self.layers = clones(layer, N)
    self.norm = nn.LayerNorm(layer.size)

  def forward(self, x, mask):
    "Pass the input (and mask) through each layer in turn."
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)

class Decoder(nn.Module):
  "Core encoder is a stack of N layers"
  def __init__(self, layer, N):
    super(Decoder, self).__init__()
    self.layers = clones(layer, N)
    self.norm = nn.LayerNorm(layer.size)

  def forward(self, x, memory, src_mask, tgt_mask):
    "Pass the input (and mask) through each layer in turn."
    for layer in self.layers:
      x = layer(x, memory, src_mask, tgt_mask)
    return self.norm(x)

class SublayerConnection(nn.Module):
  """
  A residual connection followed by a layer norm.
  Note for code simplicity the norm is first as opposed to last.
  """

  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = nn.LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    "Apply residual connection to any sublayer with the same size."
    # Post-LN applied according to best modern practices. Attention is all you need paper applies Pre-LN.
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

class DecoderLayer(nn.Module):
  "Encoder is made up of self-attn, src-attn and feed forward (defined below)"
  def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
    super(DecoderLayer, self).__init__()
    self.self_attn = self_attn
    self.src_attn = src_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SublayerConnection(size, dropout), 3)
    self.size = size

  def forward(self, x, memory, src_mask, tgt_mask):
    "Follow Figure 1 (right) for connections."
    m = memory
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # the 'x's are query, key, and value for the target sequence
    x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
    return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
  "Mask out subsequent positions"
  attn_shape = (1, size, size)
  mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).bool()
  return mask

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
      # Unsqueeze to prepare for broadcasting, e.g., (N, S) -> (N, 1, S)
      mask = mask.unsqueeze(1)
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

def make_model(vocab_size, n_layers=NUM_LAYERS, d_model=EMBED_DIM, d_ff=HIDDEN_DIM, h=NUM_HEADS, dropout=0.1):
  "Helper: Construct a model from hyperparameters."
  c = copy.deepcopy
  attn = MultiHeadedAttention(h, d_model, dropout)
  ff = PositionwiseFeedForward(d_model, d_ff, dropout)
  position = PositionalEncoding(d_model, dropout, max_len=MAX_LEN)
  # The generator projects the decoder's output to the vocabulary size
  generator = Generator(d_model, vocab_size)
  # The source and target embeddings can be shared in tasks like this
  embedding = nn.Embedding(vocab_size, d_model)

  model = EncoderDecoder(
    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_layers),
    Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_layers),
    nn.Sequential(c(embedding), c(position)), # Source embedding
    nn.Sequential(c(embedding), c(position)), # Target embedding
    generator
    )

  # Initialize parameters with Glorot / fan_avg.
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  return model.to(device)
    
def train_one_epoch_generative(model, dataloader, loss_fn, optimizer, epoch):
  model.train()
  total_loss = 0
  log_interval = 100
  start_time = time.time()

  for i, (src, tgt) in enumerate(dataloader):
    # Prepare target input and output sequences
    tgt_input = tgt[:, :-1]  # All but the last token
    tgt_output = tgt[:, 1:]   # All but the first token (SOS)

    # Create masks
    # Source mask is not strictly needed as src is fixed size (1), but good practice
    src_mask = (src == PAD_IDX).unsqueeze(1)
    tgt_pad_mask = (tgt_input == PAD_IDX).unsqueeze(1)
    look_ahead_mask = subsequent_mask(tgt_input.size(1))
    combined_mask = tgt_pad_mask | look_ahead_mask

    # Forward pass
    optimizer.zero_grad()
    # The model returns the decoder's output, which has shape (batch, seq_len, d_model)
    decoder_output = model(src, tgt_input, src_mask=src_mask, tgt_mask=combined_mask)
    # The generator projects this to log-probabilities over the vocabulary
    log_probs = model.generator(decoder_output)
    
    # We need to reshape the output and target for the loss function
    loss = loss_fn(
        log_probs.contiguous().view(-1, VOCAB_SIZE),
        tgt_output.contiguous().view(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    total_loss += loss.item()

    if i % log_interval == 0 and i > 0:
      cur_loss = total_loss / log_interval
      elapsed = time.time() - start_time
      print(f'| epoch {epoch:3d} | {i:5d}/{len(dataloader):5d} batches | '
            f'loss {cur_loss:5.2f} | perplexity {math.exp(cur_loss):8.2f}')
      total_loss = 0
      start_time = time.time()

def greedy_decode(model, src, max_len, start_symbol):
  model.eval()
  src_mask = (src == PAD_IDX).unsqueeze(1)
  memory = model.encode(src, src_mask=src_mask)
  ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
  for i in range(max_len - 1):
    tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(device)
    out = model.decode(memory, src_mask=src_mask, tgt=ys, tgt_mask=tgt_mask)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()
    ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).to(device).fill_(next_word)], dim=1)
    if next_word == EOS_IDX:
      break
  return ys

def generate_review(model, sentiment_token_idx, sentiment_str):
  print(f"\n--- Generating a {sentiment_str} review ---")
  model.eval()
  src = torch.tensor([[sentiment_token_idx]], device=device)
  generated_indices = greedy_decode(model, src, max_len=MAX_LEN, start_symbol=SOS_IDX)
  generated_tokens = vocab.get_itos()
  review_tokens = [generated_tokens[i] for i in generated_indices[0].cpu().numpy()]
  
  # Filter out special tokens for cleaner output
  review_tokens = [t for t in review_tokens if t not in ['<sos>', '<eos>', '<pad>']]
  print(" ".join(review_tokens))

if __name__ == '__main__':
  train_iter = IMDB(split='train')
  train_dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch_generative)

  model = make_model(VOCAB_SIZE)

  # Define loss function and optimizer
  criterion = nn.NLLLoss(ignore_index=PAD_IDX)
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

  generate_review(model, POSITIVE_IDX, "positive")
  generate_review(model, NEGATIVE_IDX, "negative")

  for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    train_one_epoch_generative(model, train_dataloader, criterion, optimizer, epoch)
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s |')
    print('-' * 89)

  generate_review(model, POSITIVE_IDX, "positive")
  generate_review(model, NEGATIVE_IDX, "negative")