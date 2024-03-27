import random
import numpy as np
from tinygrad import Tensor, dtypes, Device
import chess.pgn
import itertools as it
import math
import random

filename_fen = "data/dataset_2m"
pair_count = 400_000

def get_data_count():
  return np.load(f"{filename_fen}_X.npy", mmap_mode='c').shape[0]

def load_wins_loses(chunk_idx, chunk_size):
  print(f"loading chunk {chunk_idx} ({chunk_size*(chunk_idx+1)})")
  X_on_disk = np.load(f"{filename_fen}_X.npy", mmap_mode='c')
  Y_on_disk = np.load(f"{filename_fen}_Y.npy", mmap_mode='c')
  X = X_on_disk[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
  Y = Y_on_disk[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
  wins, loses = X[Y == 1], X[Y == 0]
  return wins, loses

def generate_new_pairs(wins, loses):
  x, y = _generate_new_pairs(wins, loses)
  ratio = 0.8
  X_train, X_test = x[:int(len(x)*ratio)], x[int(len(x)*ratio):]
  Y_train, Y_test = y[:int(len(y)*ratio)], y[int(len(y)*ratio):]
  X_train = Tensor(X_train, dtype=dtypes.float32, device=Device.DEFAULT)
  X_test = Tensor(X_test, dtype=dtypes.float32, device=Device.DEFAULT)
  Y_train = Tensor(Y_train, dtype=dtypes.float32, device=Device.DEFAULT)
  Y_test = Tensor(Y_test, dtype=dtypes.float32, device=Device.DEFAULT)
  return X_train, Y_train, X_test, Y_test

def _generate_new_pairs(wins, loses):
  n, k = min(len(wins), len(loses)), 2 
  assert math.comb(n, k) > pair_count, f"{len(wins)=} {len(loses)=}"
  x, y = np.empty((pair_count, 2, 773)), np.empty((pair_count, 2))
  batch_size = 1
  for i in range(pair_count//batch_size):
    wins_batch = wins[np.random.choice(wins.shape[0], size=batch_size)]
    loses_batch = loses[np.random.choice(loses.shape[0], size=batch_size)]
    start_x, end_x = i*batch_size, (i+1)*batch_size
    if random.random() < 0.5:
      x[start_x:end_x, :] = np.stack((wins_batch, loses_batch), axis=1)
      y[start_x:end_x, :] = np.full((batch_size, 2), [1, 0])
    else:
      x[start_x:end_x, :] = np.stack((loses_batch, wins_batch), axis=1)
      y[start_x:end_x, :] = np.full((batch_size, 2), [0, 1])
  return x, y

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

def generate_fen_dataset(num_samples):
  game_count = 1
  X, Y = [], []

  with open(filename_pgn) as f:
    while 1:
      try:
        game = chess.pgn.read_game(f)
      except Exception:
        break

      if game is None: break
      game_count += 1
      result = game.headers["Result"]

      if result == "1/2-1/2": continue

      xs = get_random_positions(game, count=10)
      assert len(xs) == 10
      X.extend(xs)
      Y.extend([1 if result == "1-0" else 0] * 10)

      if game_count % 100 == 0:
        print(f"parsing game {game_count}, got {len(Y)} samples")

      if len(Y) >= num_samples: break
    
  X, Y = np.array(X), np.array(Y)
  np.save(f"{filename_fen}_X", X)
  np.save(f"{filename_fen}_Y", Y)
  print(f"games saved to {filename_fen}")

def get_random_positions(game, count):
  board = game.board()
  ignore_moves = 5
  positions = []

  for i, move in enumerate(game.mainline_moves()):
    # cannot be from first 5 moves
    if i < 5:
      board.push(move)
      continue

    # cannot be capture move
    if not board.is_capture(move): 
      board.push(move)
      positions.append(board)
    else:
      board.push(move)
  
  return [serialize(b) for b in random.sample(positions, count)]

if __name__ == "__main__":
  filename_pgn = "data/CCRL-4040.[1828834].pgn"
  generate_fen_dataset(1e6 * 2)