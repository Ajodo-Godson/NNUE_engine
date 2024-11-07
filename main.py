import chess


def main():
    board = chess.Board()
    # print(board)
    # print(board.unicode())

    print(board.pawns)
    print(board.knights)
    print(board.bishops)
    print(board.rooks)
    print(board.queens)
    print(board.kings)


if __name__ == "__main__":
    main()
