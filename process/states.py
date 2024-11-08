import chess
import numpy as np

board = chess.Board()

NUM_SQUARES = 64
PROMOTION_PIECES = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
NUM_PROMOTIONS = len(PROMOTION_PIECES)

# Total size of the action space
ACTION_SPACE_SIZE = NUM_SQUARES * NUM_SQUARES * NUM_PROMOTIONS


# Function to map a move to an index
def move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    # Get promotion index
    if move.promotion in PROMOTION_PIECES:
        promotion_index = PROMOTION_PIECES.index(move.promotion)
    else:
        promotion_index = 0  # No promotion
    index = (
        (from_square * NUM_SQUARES * NUM_PROMOTIONS)
        + (to_square * NUM_PROMOTIONS)
        + promotion_index
    )
    return index


# The reverse function to map an index to a move
def index_to_move(index):
    from_square = index // (NUM_SQUARES * NUM_PROMOTIONS)
    to_square = (index % (NUM_SQUARES * NUM_PROMOTIONS)) // NUM_PROMOTIONS
    promotion_index = index % NUM_PROMOTIONS
    promotion = PROMOTION_PIECES[promotion_index]
    return chess.Move(from_square, to_square, promotion=promotion)


# Function to create a mask of legal moves
def get_legal_moves_mask(board):
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        index = move_to_index(move)
        mask[index] = 1
    return mask


# Generate the mask for the current board state
legal_moves_mask = get_legal_moves_mask(board)


move = chess.Move.from_uci("e2e4")
index = move_to_index(move)
print(f"Move {move.uci()} is at index {index}")
print(f"Index {index} corresponds to move {index_to_move(index).uci()}")
