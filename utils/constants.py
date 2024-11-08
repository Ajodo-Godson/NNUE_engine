import chess

NUM_SQUARES = 64
PROMOTION_PIECES = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
NUM_PROMOTIONS = len(PROMOTION_PIECES)
ACTION_SPACE_SIZE = NUM_SQUARES * NUM_SQUARES * NUM_PROMOTIONS  # 64 * 64 * 5 = 20,480