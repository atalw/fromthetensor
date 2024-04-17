import random
import numpy as np
from tinygrad import Tensor, dtypes, Device
import chess.pgn
import itertools as it
import math
import random

filename_fen = "data/dataset_2m"
pair_count = 500_000

def get_data_count():
  return np.load(f"{filename_fen}_Y.npy", mmap_mode='c').shape[0]

def load_wins_loses(chunk_idx, chunk_size):
  print(f"loading chunk {chunk_idx} ({chunk_size*(chunk_idx+1)})")
  X_on_disk = np.load(f"{filename_fen}_X.npy", mmap_mode='c')
  Y_on_disk = np.load(f"{filename_fen}_Y.npy", mmap_mode='c')
  X = X_on_disk[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
  Y = Y_on_disk[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]
  wins, loses = X[Y == 1], X[Y == 0]
  return wins, loses

def generate_new_pairs(wins, loses):
  x1, x2, y = _generate_new_pairs(wins, loses)
  ratio = 0.8
  X1_train, X1_test = x1[:int(len(x1)*ratio)], x1[int(len(x1)*ratio):]
  X2_train, X2_test = x2[:int(len(x2)*ratio)], x2[int(len(x2)*ratio):]
  Y_train, Y_test = y[:int(len(y)*ratio)], y[int(len(y)*ratio):]
  X1_train = Tensor(X1_train, dtype=dtypes.float32, device=Device.DEFAULT)
  X2_train = Tensor(X2_train, dtype=dtypes.float32, device=Device.DEFAULT)
  X1_test = Tensor(X1_test, dtype=dtypes.float32, device=Device.DEFAULT)
  X2_test = Tensor(X2_test, dtype=dtypes.float32, device=Device.DEFAULT)
  Y_train = Tensor(Y_train, dtype=dtypes.float32, device=Device.DEFAULT)
  Y_test = Tensor(Y_test, dtype=dtypes.float32, device=Device.DEFAULT)
  return X2_train, X2_train, Y_train, X1_test, X2_test, Y_test

def _generate_new_pairs(wins, loses):
  # n, k = min(len(wins), len(loses)), 2 
  # assert math.comb(n, k) > pair_count, f"{len(wins)=} {len(loses)=}"
  x1, x2, y = [], [], []
  for i in range(pair_count):
    win = wins[np.random.choice(wins.shape[0])]
    loss = loses[np.random.choice(loses.shape[0])]
    if random.random() < 0.5:
      # NOTE: list append is faster than np append apparently
      x1.append(win)
      x2.append(loss)
      y.append([1, 0])
    else:
      x1.append(loss)
      x2.append(win)
      y.append([0, 1])
  return x1, x2, y

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