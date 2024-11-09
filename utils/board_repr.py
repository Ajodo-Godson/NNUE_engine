import torch
import chess


def board_to_input_tensor(board):
    tensor = torch.zeros(16, 8, 8, dtype=torch.float32)

    piece_type_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Map white and black pieces
    for piece_type, plane in piece_type_to_plane.items():
        # White pieces
        for square in board.pieces(piece_type, chess.WHITE):
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[plane, row, col] = 1.0
        # Black pieces
        for square in board.pieces(piece_type, chess.BLACK):
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[plane + 6, row, col] = 1.0

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
    tensor[15].fill_(float(board.turn))

    # Flatten the tensor
    tensor = tensor.view(-1)
    return tensor
