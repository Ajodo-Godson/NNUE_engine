import chess
from collections import defaultdict

board = chess.Board()

PIECES_PLACEMENT = defaultdict(list)


def convert_bit_to_board(bit):
    arr = []
    for i in range(0, 64, 8):
        arr.append(bit[i : i + 8])
    return arr


# So the chess pieces are bitencoded.
# These bit encodings basically tell us where the pieces are on the board

# Let's convert the bit encodiogn to a standard chess board placement of 8 by 8
bit_conv = bin(board.pawns)[2:].zfill(64)

for squares in chess.PIECE_NAMES:
    if squares is not None:
        piece_att = getattr(board, squares + "s")

        PIECES_PLACEMENT[squares] = convert_bit_to_board(bin(piece_att)[2:].zfill(64))
