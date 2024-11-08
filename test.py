import chess
from utils.board_repr import board_to_input_tensor

# Initialize a board with a custom position (optional)
board = chess.Board()

# Convert the board to input tensor
input_tensor = board_to_input_tensor(board)

# Check the shape of the tensor
print(f"Input tensor shape: {input_tensor.shape}")  # Should be (1024,)

# Optionally, reshape and visualize the tensor
tensor_reshaped = input_tensor.view(16, 8, 8)
print("Tensor channels:")
for i in range(16):
    print(f"Channel {i}:")
    print(tensor_reshaped[i])
