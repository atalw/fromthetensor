import random
import math
import numpy as np
import chess.pgn

filename = "data/CCRL-4040.[1828834].pgn"

def get_random_positions(game, white_win, count):
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
  
  return [serialize(b) for b in random.sample(positions, count)], [int(white_win)]*count

# convert fen to bitboard
def serialize(board):
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
  
def get_dataset(n):
  count = 0
  X, Y = [], []

  with open(filename) as f:
    while 1:
      try:
        game = chess.pgn.read_game(f)
      except Exception:
        break

      if game is None: break
      result = game.headers["Result"]
      if result == "1-0": # white wins
        xs, ys = get_random_positions(game, white_win=True, count=10)
        X.extend(xs)
        Y.extend(ys)
      elif result == "0-1": # white loses
        xs, ys = get_random_positions(game, white_win=False, count=10)
        X.extend(xs)
        Y.extend(ys)
      else: # ignore draw
        pass

      if count % 100:
        print(f"parsing game {count}, got {len(X)} samples")

      count += 1
      if count >= n: break
    
  return np.array(X), np.array(Y)
  
if __name__ == "__main__":
  X, Y = get_dataset(1e5)
  np.savez("data/dataset_100k.npz", X, Y)