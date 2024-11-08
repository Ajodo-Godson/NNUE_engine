from main import generate_dataset


def test_generate_dataset():
    num_positions = 100
    stockfish_path = "stockfish"
    save_path = "test_dataset.pth"
    generate_dataset(num_positions, stockfish_path, save_path)


test_generate_dataset()
