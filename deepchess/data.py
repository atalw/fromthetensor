import random
import numpy as np
from tinygrad import Tensor, dtypes, Device
import chess.pgn
import math
import random

filename_fen = "data/dataset"
# pair_count = 500

def load_wins_loses():
  wins_on_disk = np.load(f"{filename_fen}_wins.npy", mmap_mode='c')
  loses_on_disk = np.load(f"{filename_fen}_loses.npy", mmap_mode='c')
  return wins_on_disk, loses_on_disk

def get_data_count():
  wins, loses = load_wins_loses()
  return wins.shape[0] + loses.shape[0]

def _generate_new_pairs(wins, loses, pair_count):
  # store last n positions for test
  n_test = 100_000
  # indexing mmap randomly is the bottleneck, hack: choose continuous samples for fast indexing
  i, j = np.random.choice(wins.shape[0]-pair_count-n_test), np.random.choice(loses.shape[0]-pair_count-n_test)
  # moves data to gpu
  win_samples, loss_samples = Tensor(wins[i:i+pair_count]), Tensor(loses[j:j+pair_count])
  shuffle1, shuffle2 = Tensor.randint(pair_count, high=pair_count), Tensor.randint(pair_count, high=pair_count)
  win_samples, loss_samples = win_samples[shuffle1], loss_samples[shuffle2]
  # tensor puzzles ftw
  conditions =  Tensor.rand((pair_count, 1)) <= 0.5
  x1 = Tensor.where(conditions, win_samples, loss_samples)
  x2 = Tensor.where(conditions, loss_samples, win_samples)
  y = Tensor.where(conditions, Tensor([[1.0, 0.0]]), Tensor([[0.0, 1.0]]))
  # sanity
  assert len(x1.shape) == len(x2.shape) == len(y.shape) == 2
  assert x1.shape[0] == x2.shape[0] == y.shape[0] == pair_count
  assert x1.shape[1] == x2.shape[1] == 773, f"{x1.shape=}, {x2.shape=}"
  assert y.shape[1] == 2
  return x1, x2, y

def generate_new_pairs(wins, loses, with_test=False):
  x1, x2, y = _generate_new_pairs(wins, loses)
  if not with_test: return x1, x2, y, None, None, None
  ratio = 0.8
  s1, s2 = math.ceil(x1.shape[0]*ratio), math.ceil(x1.shape[0]*(1-ratio))
  x1_train, x1_test = x1.split([s1, s2])
  x2_train, x2_test = x2.split([s1, s2])
  y_train, y_test = y.split([s1, s2])
  return x1_train, x2_train, y_train, x1_test, x2_test, y_test

def load_new_pairs(i):
  f = np.load(f"data/pairs/pairs_{i}")
  x1, x2, y = f['x1'], f['x2'], f['y']
  return Tensor(x1), Tensor(x2), Tensor(y)

# generate pairs and store on disk before training
def preprocess_pairs(pair_count):
  wins, loses = load_wins_loses()
  for i in range(1000):
    print(f"processing set {i}")
    x1, x2, y = _generate_new_pairs(wins, loses, pair_count)
    x1, x2, y = x1.numpy(), x2.numpy(), y.numpy()
    np.savez_compressed(f"data/pairs/pairs_{i}", x1=x1, x2=x2, y=y)

# convert fen to bitboard
def serialize(board: chess.Board):
  mapping = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
  # 2 sides, 6 pieces, 64 squares = 2*6*64 = 768 bits
  # 5 extra bits -> 1 bit side to move, 4 bits for castling rights = 773 bits
  bitboard = np.zeros(2*6*64+5, dtype=int)
  for i in range(64):
    if board.piece_at(i):
      piece= board.piece_at(i)
      color = int(piece.color) + 1
      bitboard[(mapping[piece.symbol().lower()] + i * color)] = 1
  
  bitboard[-1] = int(board.turn)
  bitboard[-2] = int(board.has_kingside_castling_rights(True))
  bitboard[-3] = int(board.has_kingside_castling_rights(False))
  bitboard[-4] = int(board.has_queenside_castling_rights(True))
  bitboard[-5] = int(board.has_queenside_castling_rights(False))

  return bitboard

def get_random_positions(game, count):
  board = game.board()
  ignore_moves = 5
  positions = []

  for i, move in enumerate(game.mainline_moves()):
    # cannot be from first 5 moves
    if i < ignore_moves:
      board.push(move)
      continue

    # cannot be capture move
    if not board.is_capture(move): 
      board.push(move)
      positions.append(board)
    else:
      board.push(move)
  
  return [serialize(b) for b in random.sample(positions, count)]

def generate_fen_dataset(num_samples):
  game_count = 1
  wins, loses, both = [], [], [] # from white's perspective

  with open(filename_pgn) as f:
    while 1:
      try: game = chess.pgn.read_game(f)
      except Exception: break
      if game is None: continue 

      result = game.headers["Result"]
      if result == "1/2-1/2": continue
      xs = get_random_positions(game, count=10)
      if result == "1-0": wins.extend(xs)
      else: loses.extend(xs)
      both.extend(xs)

      game_count += 1
      if game_count % 1000 == 0:
        print(f"parsing game {game_count}, got {len(both)} samples")
      if len(both) >= num_samples: break
    
  nwins, nloses, nboth = np.array(wins), np.array(loses), np.array(both)
  np.save(f"{filename_fen}_wins", nwins)
  np.save(f"{filename_fen}_loses", nloses)
  np.save(f"{filename_fen}_combined", nboth)
  print(f"games saved to {filename_fen}")

if __name__ == "__main__":
  filename_pgn = "data/CCRL-4040.[1828834].pgn"
  # generate_fen_dataset(1e6 * 2)
  preprocess_pairs(600_000)
  # preprocess_pairs(10)