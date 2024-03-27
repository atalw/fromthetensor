import models.distilled as distilled_model
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import Tensor
import copy
import chess
import data
import numpy as np

def evaluate_pos(x, y):
  x_bitboard, y_bitboard = data.serialize(x), data.serialize(y)
  input = Tensor(np.concatenate((x_bitboard, y_bitboard), axis=-1))
  out = model(input)
  # print(out.shape, out.numpy())
  return (x, y) if out[0].numpy() > out[1].numpy() else (y, x)

# position and value cache
board_v_cache = {}

# traditional alphabeta algorithm stores values in alpha and beta
# we are storing positions instead (refer paper)
# fail-soft alpha-beta
def alphabeta(board: chess.Board, depth, alpha: chess.Board, beta: chess.Board, maximizing_player) -> chess.Board:
  if depth == 0 or board.is_game_over():
    return board 
  if maximizing_player:
    v = None
    for move in board.legal_moves:
      next_board = copy.copy(board)
      next_board.push(move)

      if next_board.fen() in board_v_cache: return board_v_cache[next_board.fen()]

      if v == None: v = alphabeta(next_board, depth-1, alpha, beta, False)
      if alpha == None: alpha = v

      v = evaluate_pos(v, alphabeta(next_board, depth-1, alpha, beta, False))[0]
      board_v_cache[next_board.fen()] = v
      alpha = evaluate_pos(alpha, v)[0]

      if beta != None and evaluate_pos(v, beta)[0] == v:
        break
    return v
  else:
    v = None
    for move in board.legal_moves:
      next_board = copy.copy(board)
      next_board.push(move)

      if next_board.fen() in board_v_cache: return board_v_cache[next_board.fen()]

      if v == None: v = alphabeta(next_board, depth-1, alpha, beta, True)
      if beta == None: beta = v

      v = evaluate_pos(v, alphabeta(next_board, depth-1, alpha, beta, True))[1]
      board_v_cache[next_board.fen()] = v
      beta = evaluate_pos(beta, v)[1]

      if alpha != None and evaluate_pos(v, alpha)[0] == v:
        break
    return v


def computer_turn(board: chess.Board):
  alpha, beta, v = None, None, None
  depth = 4
  best_move = None
  print("thinking...")

  for move in board.legal_moves:
    next_board = copy.copy(board)
    next_board.push(move)
    if v == None:
      v = alphabeta(next_board, depth-1, alpha, beta, False)
      best_move = move
      if alpha == None: alpha = v
    else:
      new_v = evaluate_pos(alphabeta(next_board, depth-1, alpha, beta, False), v)[0]
      if new_v != v:
        best_move = move
        v = new_v
      alpha = evaluate_pos(alpha, v)[0]
  print(f"best move: {best_move}")
  board.push(best_move)
  return board

def player_turn(board: chess.Board):
  while 1:
    try:
      move = input("Enter move: ")
      board.push_san(move)
      break
    except ValueError:
      print("illegal move, try again")
  
  return board


if __name__ == "__main__":
  model = distilled_model.Distilled()
  load_state_dict(model, safe_load("./ckpts/distilled_1m_final_epoch_850.safe"))
  # load_state_dict(model, safe_load("./ckpts/distilled.safe"))

  n = 0
  board = chess.Board()
  while not board.is_game_over():
    print(board)
    print(board.fen())
    print()
    if n % 2 == 0: board = computer_turn(board)
    else: board = player_turn(board)
    n += 1
  print(board)
  print("game over")

