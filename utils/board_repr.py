import torch
import chess


def board_to_input_tensor(board):
    # Initialize a tensor of zeros
    tensor = torch.zeros(16, 8, 8, dtype=torch.float32)  # 16 channels, 8x8 board

    # Mapping from piece type to plane index
    piece_type_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Place pieces on the tensor
    for piece_type in piece_type_to_plane.keys():
        # White pieces
        white_bitboard = board.pieces(piece_type, chess.WHITE)
        white_indices = white_bitboard
        white_plane = piece_type_to_plane[piece_type]
        for square in chess.SquareSet(white_indices):
            row = 7 - chess.square_rank(square)  # Adjust for correct orientation
            col = chess.square_file(square)
            tensor[white_plane, row, col] = 1.0

        # Black pieces
        black_bitboard = board.pieces(piece_type, chess.BLACK)
        black_indices = black_bitboard
        black_plane = piece_type_to_plane[piece_type] + 6  # Offset for black pieces
        for square in chess.SquareSet(black_indices):
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[black_plane, row, col] = 1.0

    # Empty squares
    occupied = board.occupied
    empty_squares = ~occupied & chess.BB_ALL
    for square in chess.SquareSet(empty_squares):
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        tensor[12, row, col] = 1.0

    # Castling rights
    tensor[13].fill_(float(board.has_kingside_castling_rights(board.turn)))
    tensor[14].fill_(float(board.has_queenside_castling_rights(board.turn)))

    # Side to move
    tensor[15].fill_(float(board.turn))  # 1.0 for White, 0.0 for Black

    tensor = tensor.view(-1)

    return tensor
