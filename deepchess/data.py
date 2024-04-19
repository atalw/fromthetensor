import random
import numpy as np
from tinygrad import Tensor, dtypes
import chess.pgn
import random

filename_fen = "data/dataset"

def load_wins_loses(mmap=True):
  wins = np.load(f"{filename_fen}_wins.npy", mmap_mode='c' if mmap else None)
  loses = np.load(f"{filename_fen}_loses.npy", mmap_mode='c' if mmap else None)
  return wins, loses

def load_new_pairs(i):
  f = np.load(f"data/pairs/pairs_{i}.npz", mmap_mode='c')
  x1, x2, y = f['x1'], f['x2'], f['y']
  return Tensor(x1, dtype=dtypes.float32), Tensor(x2, dtype=dtypes.float32), Tensor(y, dtype=dtypes.float32) # move to gpu

def _generate_new_pairs(wins, loses, pair_count):
  win_samples = wins[np.random.randint(0, wins.shape[0], pair_count)]
  loss_samples = loses[np.random.randint(0, loses.shape[0], pair_count)]
  # tensor puzzles ftw
  conditions =  np.random.rand(pair_count, 1) <= 0.5
  x1 = np.where(conditions, win_samples, loss_samples)
  x2 = np.where(conditions, loss_samples, win_samples)
  y = np.where(conditions, np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
  # sanity
  assert len(x1.shape) == len(x2.shape) == len(y.shape) == 2
  assert x1.shape[0] == x2.shape[0] == y.shape[0] == pair_count
  assert x1.shape[1] == x2.shape[1] == 773, f"{x1.shape=}, {x2.shape=}"
  assert y.shape[1] == 2
  return x1, x2, y

def generate_test_set():
  wins, loses = load_wins_loses(mmap=True)
  n_test = 100_000
  wins, loses = wins[n_test:], loses[n_test:]
  x1, x2, y = _generate_new_pairs(wins, loses, n_test)
  return Tensor(x1, dtype=dtypes.float32), Tensor(x2, dtype=dtypes.float32), Tensor(y, dtype=dtypes.float32)

# generate pairs and store on disk before training
def preprocess_pairs(pair_count):
  wins, loses = load_wins_loses(mmap=False)
  n_test = 100_000 # keep last n positions for test set
  wins, loses = wins[:-n_test], loses[:-n_test]
  for i in range(1000):
    print(f"processing set {i}")
    x1, x2, y = _generate_new_pairs(wins, loses, pair_count)
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