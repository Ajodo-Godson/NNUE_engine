# import chess
# from utils.board_repr import board_to_input_tensor
# import torch


# # Initialize a board with a custom position (optional)
# board = chess.Board()

# # Convert the board to input tensor
# input_tensor = board_to_input_tensor(board)

# # Check the shape of the tensor
# print(f"Input tensor shape: {input_tensor.shape}")  # Should be (1024,)

# # Optionally, reshape and visualize the tensor
# tensor_reshaped = input_tensor.view(16, 8, 8)
# # print("Tensor channels:")
# # for i in range(16):
# #     print(f"Channel {i}:")
# #     print(tensor_reshaped[i])


# model = torch.load("data/chess_dataset.pt")
# print(model)
# for param in model:
#     print(param)
import zipfile

zip_path = "data/datasets.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    print("Files in the zip archive:")
    print(z.namelist())
