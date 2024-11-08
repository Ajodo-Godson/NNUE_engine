import chess
from utils.constants import (
    NUM_SQUARES,
    PROMOTION_PIECES,
    NUM_PROMOTIONS,
)


def move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square
    promotion_piece = move.promotion
    promotion_index = (
        PROMOTION_PIECES.index(promotion_piece)
        if promotion_piece in PROMOTION_PIECES
        else 0
    )
    index = (
        (from_square * NUM_SQUARES * NUM_PROMOTIONS)
        + (to_square * NUM_PROMOTIONS)
        + promotion_index
    )
    return index


def index_to_move(index):
    from_square = index // (NUM_SQUARES * NUM_PROMOTIONS)
    to_square = (index % (NUM_SQUARES * NUM_PROMOTIONS)) // NUM_PROMOTIONS
    promotion_index = index % NUM_PROMOTIONS
    promotion_piece = PROMOTION_PIECES[promotion_index]
    return chess.Move(from_square, to_square, promotion=promotion_piece)
