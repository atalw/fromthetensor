import random
import math
import numpy as np
import chess.pgn

filename = "data/CCRL-4040.[1828834].pgn"

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
      positions.append(board.fen())
    else:
      board.push(move)
  
  return [serialize(fen) for fen in random.sample(positions, count)]

# convert fen to bitboard
def serialize(fen):
  mapping = {
      'p': 0,
      'n': 1,
      'b': 2,
      'r': 3,
      'q': 4,
      'k': 5,
      'P': 6,
      'N': 7,
      'B': 8,
      'R': 9,
      'Q': 10,
      'K': 11
  }

  # 2 sides, 6 pieces, 64 squares = 2*6*64 = 768 bits
  # 5 extra bits -> 1 bit side to move, 4 bits for castling rights
  bitboard = np.zeros((1, 773), dtype=int)
  i = 0
  [position, turn, castling, _, _, _] = fen.split(' ')

  for c in position:
    if c == '/':
      continue
    elif c >= '1' and c <= '8':
      i += (ord(c) - ord('0')) * 12
    else:
      bitboard[0, i+mapping[c]] = 1
      i += 12
  bitboard[0, 768] = 1 if turn == 'w' else 0
  bitboard[0, 769] = 1 if 'K' in castling else 0
  bitboard[0, 770] = 1 if 'Q' in castling else 0
  bitboard[0, 771] = 1 if 'k' in castling else 0
  bitboard[0, 772] = 1 if 'q' in castling else 0
  return bitboard
  
def get_dataset(n):
  count = 0
  X_win, X_lose = [], []

  with open(filename) as f:
    while 1:
      try:
        game = chess.pgn.read_game(f)
      except Exception:
        break

      if game is None: break
      result = game.headers["Result"]
      if result == "1-0": # white wins
        X_win.extend(get_random_positions(game, 10))
      elif result == "0-1": # white loses
        X_lose.extend(get_random_positions(game, 10))
      # ignore draw

      if count % 100:
        print(f"parsing game {count}, got {len(X_win)+len(X_lose)} samples")

      count += 1
      if count >= n: break
  
  test_ratio = 0.2
  index_win = math.floor(len(X_win)*(1-test_ratio))
  X_win, Y_win= X_win[:index_win], X_win[index_win:]
  index_win = math.floor(len(X_lose)*(1-test_ratio))
  X_lose, Y_lose= X_lose[:index_win], X_lose[index_win:]

  return np.array(X_win), np.array(X_lose), np.array(Y_win), np.array(Y_lose)

if __name__ == "__main__":

  X_win, X_lose, Y_win, Y_lose = get_dataset(1e6)
  print(f"{X_win.shape[0]+Y_win.shape[0]} white win positions, {X_lose.shape[0]+Y_lose.shape[0]} white lose positions")

  np.savez("data/x_win_1m.npz", X_win, Y_win)
  np.savez("data/x_lose_1m.npz", X_lose, Y_lose)