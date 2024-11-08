import torch
import numpy as np
import chess
from utils.constants import ACTION_SPACE_SIZE
from utils.move_mapping import move_to_index


def get_legal_moves_mask(board):
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        index = move_to_index(move)
        mask[index] = 1.0
    return torch.tensor(mask)
