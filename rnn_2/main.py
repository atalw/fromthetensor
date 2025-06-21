import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math
import os

def read_music():
  current_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(current_dir, "french_music.txt")
  try:
    with open(file_path, 'r', encoding='latin-1') as f:
      music_data = f.read()
      return music_data
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return [], 0
  except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    return [], 0

def create_music_vocabulary(data):
  all_characters = sorted(list(set(data)))
  vocab_size = len(all_characters)

  print(f"Vocabulary contains {vocab_size} unique characters.")
  print(f"Vocabulary: {''.join(all_characters)}")

  # Create character-to-index and index-to-character mappings
  char_to_idx = {char: i for i, char in enumerate(all_characters)}
  idx_to_char = {i: char for i, char in enumerate(all_characters)}

  return char_to_idx, idx_to_char

if __name__ == "__main__":
  data = read_music()
  char_to_idx, idx_to_char = create_music_vocabulary(data)
  print(char_to_idx)
  print(idx_to_char)

