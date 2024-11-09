import torch
import numpy as np
import chess


def board_to_input_tensor(board):
    piece_to_int = {
        "P": 1,
        "N": 2,
        "B": 3,
        "R": 4,
        "Q": 5,
        "K": 6,
        "p": -1,
        "n": -2,
        "b": -3,
        "r": -4,
        "q": -5,
        "k": -6,
        None: 0,
    }
    board_tensor = np.zeros((8, 8), dtype=np.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        row = 7 - (square // 8)
        col = square % 8
        piece_symbol = piece.symbol() if piece else None
        board_tensor[row, col] = piece_to_int.get(piece_symbol, 0)
    # Convert to PyTorch tensor
    board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
    # Add channel dimension for convolutional layers (shape: [1, 8, 8])
    board_tensor = board_tensor.unsqueeze(0)
    return board_tensor
